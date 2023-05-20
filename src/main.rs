use anyhow::Result;
use async_openai::types::ChatCompletionRequestMessage;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::Redirect;
use axum::routing::post;
use gpt_rs::history::{History, Info, InfoBuilder, Message};
use gpt_rs::websocket::WebSocket;
use gpt_rs::{DATA_DIR, MAX_TOKENS, RESPONSE_SIZE};
use std::io;
use std::sync::Arc;
use std::{fs::File, io::Write};

use axum::{response::IntoResponse, routing::get, Router};

use std::borrow::Cow;
use std::ops::ControlFlow;
use std::{net::SocketAddr, path::PathBuf};
use tower_http::{
    services::ServeDir,
    trace::{DefaultMakeSpan, TraceLayer},
};

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

//allows to extract the IP of connecting user
use axum::extract::connect_info::ConnectInfo;
use axum::extract::ws::{CloseFrame, WebSocket as AxumWebSocket, WebSocketUpgrade};

//allows to split the websocket stream into separate TX and RX branches
//use futures::{sink::SinkExt, stream::StreamExt};
use axum_sessions::{
    async_session::MemoryStore, extractors::WritableSession, PersistencePolicy, SessionLayer,
};

use gpt_rs::embeddings::Embeddings;
use gpt_rs::html::{HtmlTemplate, IndexTemplate, Message as HTMLMsg};
use gpt_rs::openai::Client;

pub struct AppState {
    embeddings: Embeddings,
    client: Client,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "example_websockets=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let store = async_session::CookieStore::new();
    let secret = b"593jfdslgdsgdssjgdsghljfshp[jmvadlk;hgadljgdahm'dvahfdlfgadssmlf"; // MUST be at least 64 bytes!
    let session_layer = SessionLayer::new(store, secret);

    let api_key =
        std::env::var("OPENAI_API_KEY").expect("Expect OPENAI_API_KEY environment variable");
    let file = File::open("./embeddings.csv").unwrap();
    let reader = std::io::BufReader::new(file);
    let embeddings = Embeddings::load(reader)?;
    let client = Client::new(&api_key);
    let data_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");

    println!("\x1b[0;32m started \x1b[0m");

    let app_state = Arc::new(AppState { embeddings, client });
    let app = Router::new()
        .route("/", get(index))
        .route("/clear_history", post(clear_history))
        .route("/websocket", get(websocket_handler))
        .nest_service("/context", ServeDir::new(DATA_DIR))
        .layer(session_layer)
        .with_state(app_state);

    axum::Server::bind(&"0.0.0.0:5000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();

    Ok(())
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    mut session: WritableSession,
) -> impl IntoResponse {
    println!("\x1b[0;32m open socket \x1b[0m");

    let eee = session.get::<String>("hist");

    println!("\x1b[0;36m eee \x1b[0m= {:?}", eee);

    let history = session
        .get::<String>("hist")
        .and_then(|filename| {
            History::load(&filename)
                .map_err(|e| println!("Couldn't open file {}: {}", filename, e))
                .ok()
        })
        .unwrap_or_else(History::new);

    // if let Some(name) = &history.name {
    //     session.insert_raw("hist", name.to_string());
    // }

    ws.on_upgrade(|socket| websocket(socket, state, history))
}

async fn websocket(mut socket: AxumWebSocket, state: Arc<AppState>, mut history: History<'_>) {
    //send a ping (unsupported by some browsers) just to kick things off and get a response
    //

    println!("\x1b[0;32m open socket2 \x1b[0m");
    match WebSocket::initiate(socket).await {
        Err(e) => {
            println!(" Couldn't initiate websocket {}", e);
            return;
        }
        Ok(mut socket) => {
            while let Some(msg) = socket.next().await {
                println!("Got message: {}", msg);
                if let Err(e) = process_message(
                    &msg,
                    &mut history,
                    &state.embeddings,
                    &state.client,
                    &mut socket,
                )
                .await
                {
                    println!("Got error {} while processing message {}", e, msg);
                }
            }
        }
    }
}

async fn process_message<'a, 'b>(
    msg: &str,
    history: &mut History<'b>,
    embeddings: &'a Embeddings,
    client: &Client,
    socket: &mut WebSocket,
) -> Result<()>
where
    'a: 'b,
{
    let mut info = InfoBuilder::default();
    let user_msg = Message::user(&msg)?;
    info.user_message_tokens(user_msg.tokens.into());
    socket.send(HTMLMsg::from(&user_msg)).await?;

    history.user(user_msg.clone());

    let pruned_messages = history.prune_history();
    info.history_count(pruned_messages.len());

    let history_size = pruned_messages.iter().map(|m| m.tokens).sum::<u16>();
    info.history_size(history_size.into());

    let emb = client.get_embedding(&msg).await?;
    let (context_msg, context_info) =
        embeddings.prepare_context(&emb, MAX_TOKENS - history_size - RESPONSE_SIZE)?;

    info.context_info(context_info);

    let mut messages = vec![context_msg];
    messages.extend_from_slice(
        &pruned_messages
            .iter()
            .map(|m| m.msg.clone())
            .collect::<Vec<ChatCompletionRequestMessage>>(),
    );
    let resp = client.chat(&messages).await?;
    let resp_msg = Message::from_response(resp, info.build()?)?;
    socket.send(HTMLMsg::from(&resp_msg)).await?;
    history.assistant(resp_msg);
    Ok(())
}

async fn clear_history(mut session: WritableSession) -> impl IntoResponse {
    session.destroy();
    Redirect::to("/")
}

async fn index(mut session: WritableSession) -> impl IntoResponse {
    let history = session
        .get::<String>("hist")
        .and_then(|filename| {
            History::load(&filename)
                .map_err(|e| println!("Couldn't open file {}: {}", filename, e))
                .ok()
        })
        .unwrap_or_else(History::new);
    if let Some(hist_name) = &history.name {
        session.insert("hist", hist_name.to_string()).unwrap();
    }

    let history = history
        .messages()
        .iter()
        .map(|m| HTMLMsg::from(m))
        .collect();

    let template = IndexTemplate { history };
    HtmlTemplate(template)
}
