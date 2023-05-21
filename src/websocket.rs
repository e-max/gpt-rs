use anyhow::Result;
use serde::Serialize;
use std::ops::ControlFlow;
//use tracing::{warn,debug};
use std::println as info;
use std::println as error;
use std::println as warn;
use std::println as debug;


use axum::extract::ws::{Message as WsMessage, WebSocket as AxumWebSocket};

pub struct WebSocket {
    socket: AxumWebSocket,
}

impl WebSocket {
    pub async fn initiate(mut socket: AxumWebSocket) -> Result<Self> {
        //send a ping (unsupported by some browsers) just to kick things off and get a response
        socket.send(WsMessage::Ping(vec![1, 2, 3])).await?;
        debug!("Pinged ...");

        Ok(Self { socket })
    }

    pub async fn next(&mut self) -> Option<String> {
        while let Some(msg) = self.socket.recv().await {
            if let Ok(msg) = msg {
                match process_message(msg) {
                    ControlFlow::Continue(Some(s)) => return Some(s),
                    ControlFlow::Continue(None) => continue,
                    ControlFlow::Break(_) => return None,
                }
            } else {
                warn!("client  abruptly disconnected");
                return None;
            }
        }
        None
    }

    pub async fn send<S>(&mut self, msg: S) -> Result<()>
    where
        S: Serialize,
    {
        self.socket
            .send(WsMessage::Text(serde_json::to_string(&msg)?))
            .await?;
        Ok(())
    }
}

fn process_message(msg: WsMessage) -> ControlFlow<(), Option<String>> {
    match msg {
        WsMessage::Text(t) => {
            debug!(">>>  sent str: {:?}", t);
            ControlFlow::Continue(Some(t))
        }
        WsMessage::Binary(d) => {
            debug!(">>> sent {} bytes: {:?}", d.len(), d);
            ControlFlow::Continue(None)
        }
        WsMessage::Close(c) => {
            if let Some(cf) = c {
                warn!(
                    ">>> sent close with code {} and reason `{}`",
                    cf.code, cf.reason
                );
            } else {
                warn!(">>>  somehow sent close message without CloseFrame");
            }
            ControlFlow::Break(())
        }

        WsMessage::Pong(v) => {
            debug!(">>> sent pong with {:?}", v);
            ControlFlow::Continue(None)
        }
        // You should never need to manually handle WsMessage::Ping, as axum's websocket library
        // will do so for you automagically by replying with Pong and copying the v according to
        // spec. But if you need the contents of the pings you can see them here.
        WsMessage::Ping(v) => {
            debug!(">>>  sent ping with {:?}", v);
            ControlFlow::Continue(None)
        }
    }
}
