use anyhow::Result;
use async_openai::types::ChatCompletionRequestMessage;
use axum::extract::State;
use axum::response::Redirect;
use axum::routing::post;
use gpt_rs::history::{History, InfoBuilder, Message};
use gpt_rs::websocket::WebSocket;
use gpt_rs::cli::cli_chat_loop;
use gpt_rs::{DATA_DIR, MAX_TOKENS, RESPONSE_SIZE};
use gpt_rs::timer;
use std::fs::File;
use std::sync::Arc;

use structopt::StructOpt;
//use tracing::{info,error,warn};

use tokio::signal;



use axum::{response::IntoResponse, routing::get, Router};

use tower_http::services::ServeDir;

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use axum::extract::ws::{WebSocket as AxumWebSocket, WebSocketUpgrade};

//allows to split the websocket stream into separate TX and RX branches
//use futures::{sink::SinkExt, stream::StreamExt};
use axum_sessions::{extractors::WritableSession, SessionLayer};

use gpt_rs::embeddings::Embeddings;
use gpt_rs::html::{HtmlTemplate, IndexTemplate, Message as HTMLMsg};
use gpt_rs::openai::Client;
use tracing_subscriber::fmt::format::FmtSpan;

use std::println as info;
use std::println as error;
use std::println as warn;

pub struct AppState {
    embeddings: Embeddings,
    client: Client,
}


#[derive(Debug, StructOpt)]
#[structopt(name = "gpt-rs", about = "AI chatbot webapp")]
struct Opt {
    #[structopt(short = "l", long = "listen", default_value = "0.0.0.0:5000")]
    listen: String,

    #[structopt(short = "c", long = "cli")]
    cli: bool,
}


#[tokio::main]
async fn main() -> Result<()> {
    let opt = Opt::from_args();
    std::env::set_var("RUST_LOG", "info");


    info!("gpt-rs starting up...");

    let store = async_session::CookieStore::new();
    let secret = b"593jfdslgdsgdssjgdsghljfshp[jmvadlk;hgadljgdahm'dvahfdlfgadssmlf"; // MUST be at least 64 bytes!
    let session_layer = SessionLayer::new(store, secret);

    let api_key =
        std::env::var("OPENAI_API_KEY").expect("Expect OPENAI_API_KEY environment variable");
    let file = File::open("./embeddings.csv").unwrap();
    let reader = std::io::BufReader::new(file);
    let embeddings = Embeddings::load(reader)?;
    info!("Loaded embeddings");

    let client = Client::new(&api_key);

    if opt.cli {
        cli_chat_loop(&embeddings, &client).await;
        return Ok(())
    }

    info!("\x1b[0;32mlistening on {} \x1b[0m", opt.listen);

    let app_state = Arc::new(AppState { embeddings, client });
    let app = Router::new()
        .route("/", get(index))
        .route("/clear_history", post(clear_history))
        .route("/websocket", get(websocket_handler))
        .nest_service("/context", ServeDir::new(DATA_DIR))
        .layer(session_layer)
        .with_state(app_state);

    axum::Server::bind(&opt.listen.parse().unwrap())
        .serve(app.into_make_service())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();

    Ok(())
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    session: WritableSession,
) -> impl IntoResponse {
    let history = session
        .get::<String>("hist")
        .and_then(|filename| {
            History::load(&filename)
                .map_err(|e| error!("Couldn't open file {}: {}", filename, e))
                .ok()
        })
        .unwrap_or_else(History::new);

    // if let Some(name) = &history.name {
    //     session.insert_raw("hist", name.to_string());
    // }

    ws.on_upgrade(|socket| websocket(socket, state, history))
}

async fn websocket(socket: AxumWebSocket, state: Arc<AppState>, mut history: History<'_>) {
    //send a ping (unsupported by some browsers) just to kick things off and get a response
    //

    info!("\x1b[0;32mopen socket2 \x1b[0m");
    match WebSocket::initiate(socket).await {
        Err(e) => {
            error!("Couldn't initiate websocket {}", e);
        }
        Ok(mut socket) => {
            while let Some(msg) = socket.next().await {
                info!("Got message: {}", msg);
                if let Err(e) = process_message(
                    &msg,
                    &mut history,
                    &state.embeddings,
                    &state.client,
                    &mut socket,
                )
                .await
                {
                    warn!("Got error {} while processing message {}", e, msg);
                    for cause in e.chain() {
                        warn!("- cause {:?}", cause);
                    }
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
    let user_msg = Message::user(msg)?;
    info.user_message_tokens(user_msg.tokens.into());
    socket.send(HTMLMsg::from(&user_msg)).await?;

    history.user(user_msg.clone());

    let pruned_messages = history.prune_history();
    info.history_count(pruned_messages.len());

    let history_size = pruned_messages.iter().map(|m| m.tokens).sum::<u16>();
    info.history_size(history_size.into());

    let emb = timer!("get_embedding", {
        client.get_embedding(msg).await?
    });
    let (context_msg, context_info) = timer!("prepare_context", {
        embeddings.prepare_context(&emb, MAX_TOKENS - history_size - RESPONSE_SIZE)?
    });

    info.context_info(context_info);

    let mut messages = vec![context_msg];
    messages.extend_from_slice(
        &pruned_messages
            .iter()
            .map(|m| m.msg.clone())
            .collect::<Vec<ChatCompletionRequestMessage>>(),
    );
    let resp = timer!("openai chat completion", {
        client.chat(&messages).await?
    });
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
                .map_err(|e| error!("Couldn't open file {}: {}", filename, e))
                .ok()
        })
        .unwrap_or_else(History::new);
    if let Some(hist_name) = &history.name {
        session.insert("hist", hist_name.to_string()).unwrap();
    }

    let history = history.messages().iter().map(HTMLMsg::from).collect();

    let template = IndexTemplate { history };
    HtmlTemplate(template)
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    println!("signal received, starting graceful shutdown");
}
