use std::{
    fs::OpenOptions,
    io::{BufRead, Write},
    path::Path,
};
use tracing::error;

use anyhow::Result;
use async_openai::types::{ChatCompletionRequestMessage, ChatCompletionResponseMessage, Role};
use derive_builder::Builder;
use rand::distributions::Alphanumeric;
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use tiktoken_rs::async_openai::num_tokens_from_messages;

use crate::{embeddings::ContextInfo, CHAT_MODEL, HISTORY_DIR, MAX_HISTORY};

pub struct History<'a> {
    pub name: Option<String>,
    messages: Vec<Message<'a>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message<'a> {
    pub msg: ChatCompletionRequestMessage,
    pub tokens: u16,
    pub info: Option<Info<'a>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub struct Info<'a> {
    pub context_info: ContextInfo<'a>,
    pub user_message_tokens: u64,
    pub history_count: usize,
    pub history_size: u64,
}

impl<'a> Message<'a> {
    pub fn user(text: &str) -> Result<Self> {
        let msg = ChatCompletionRequestMessage {
            role: Role::User,
            content: text.to_string(),
            name: None,
        };
        let tokens = num_tokens_from_messages(CHAT_MODEL, &[msg.clone()])? as u16;
        Ok(Message {
            msg,
            tokens,
            info: None,
        })
    }
    pub fn from_response(resp: ChatCompletionResponseMessage, info: Info<'a>) -> Result<Self> {
        let msg = ChatCompletionRequestMessage {
            role: resp.role,
            content: resp.content,
            name: None,
        };
        let tokens = num_tokens_from_messages(CHAT_MODEL, &[msg.clone()])? as u16;
        Ok(Message {
            msg,
            tokens,
            info: Some(info),
        })
    }

    pub fn class(&self) -> &'static str {
        match self.msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        }
    }

    pub fn prefix(&self) -> &'static str {
        match self.msg.role {
            Role::User => "You: ",
            Role::Assistant => "AI: ",
            Role::System => "system: ",
        }
    }
    pub fn content(&self) -> &str {
        &self.msg.content
    }
}

impl<'a> History<'a> {
    pub fn new() -> Self {
        let name: String = thread_rng()
            .sample_iter(&Alphanumeric)
            .take(24)
            .map(char::from)
            .collect();

        History::load(&name)
            .map_err(|e| error!("Cannot create history file {}", e))
            .unwrap_or(History {
                name: None,
                messages: vec![],
            })
    }

    pub fn load(name: &str) -> Result<Self> {
        let dir = Path::new(HISTORY_DIR);
        let filename = dir.join(name);

        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&filename)?;

        let reader = std::io::BufReader::new(file);
        let mut messages: Vec<Message> = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let message: Message = serde_json::from_str(&line)?;
            messages.push(message);
        }

        Ok(Self {
            messages,
            name: Some(name.to_string()),
        })
    }

    pub fn save(&mut self, message: &Message<'a>) -> Result<()> {
        if let Some(name) = &self.name {
            let dir = Path::new(HISTORY_DIR);
            let filename = dir.join(name);
            let mut file = OpenOptions::new()
                .append(true)
                .create(true)
                .open(filename)?;

            serde_json::to_writer(&file, message)?;
            file.write_all(b"\n")?;
            file.flush()?;
        }
        Ok(())
    }

    pub fn user(&mut self, message: Message<'a>) {
        if let Err(e) = self.save(&message) {
            error!("Couldn't save history: {} file {:?}", e, self.name);
        }
        self.messages.push(message);
    }
    pub fn assistant(&mut self, message: Message<'a>) {
        if let Err(e) = self.save(&message) {
            error!("Couldn't save history: {} file {:?}", e, self.name);
        }
        self.messages.push(message);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    pub fn prune_history(&self) -> &[Message] {
        for i in self.messages.len()..0 {
            if self.messages[i..]
                .iter()
                .map(|m| m.tokens as u64)
                .sum::<u64>()
                > MAX_HISTORY.into()
            {
                return &self.messages[i + 1..];
            }
        }
        &self.messages
    }
}

impl<'a> Default for History<'a> {
    fn default() -> Self {
        Self::new()
    }
}
