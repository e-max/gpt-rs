use askama::Template;
use async_openai::types::Role;
use axum::http::StatusCode;
use axum::response::{Html, Response};

use axum::{
    extract::ws::{Message as WsMessage, WebSocket, WebSocketUpgrade},
    response::IntoResponse,
    routing::get,
    Router,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct Message {
    #[serde(rename = "type")]
    pub typ: String,
    pub prefix: String,
    pub class: String,
    pub content: String,
    pub info: String,
}

#[derive(Template)]
#[template(path = "info.html")]
pub struct Info<'a, 'b> {
    info: &'b crate::history::Info<'a>,
}

impl From<&crate::history::Message<'_>> for Message {
    fn from(value: &crate::history::Message) -> Self {
        let typ = match value.msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        }
        .to_string();

        let prefix = match value.msg.role {
            Role::User => "You: ",
            Role::Assistant => "AI: ",
            Role::System => "System: ",
        }
        .to_string();

        let class = typ.clone();
        let content = value.msg.content.clone();
        let info = value
            .info
            .as_ref()
            .map(|info| Info { info }.render().unwrap())
            .unwrap_or_default();
        Message {
            typ,
            prefix,
            class,
            content,
            info,
        }
    }
}

#[derive(Template)]
#[template(path = "index.html")]
pub struct IndexTemplate {
    pub history: Vec<Message>,
}

pub struct HtmlTemplate<T>(pub T);

impl<T> IntoResponse for HtmlTemplate<T>
where
    T: Template,
{
    fn into_response(self) -> Response {
        match self.0.render() {
            Ok(html) => Html(html).into_response(),
            Err(err) => (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Failed to render template. Error: {}", err),
            )
                .into_response(),
        }
    }
}
