use std::io::{BufRead, stdin, Write, stdout};

use anyhow::Result;
use crate::embeddings::Embeddings;
use crate::openai::Client;
use crate::timer;
//use tracing::info;
use async_openai::types::ChatCompletionRequestMessage;
use crate::history::{History, Message};
use crate::{MAX_TOKENS, RESPONSE_SIZE};

use std::println as info;


pub async fn cli_chat_loop(
    embeddings: &Embeddings,
    client: &Client,
) {
    let stdin = stdin();
    let lines = stdin.lock().lines(); // Create a handle to stdin and a stream of lines
    let mut history = History::new();

    cli_prompt();

    for line_result in lines {
        if let Ok(line) = line_result { // If the line was read successfully...
            let msg = line.trim();
            if line == "reset" {
                history = History::new();
                println!("History was reset");
            } else {
                let response = cli_process_message(&msg, embeddings, client, &mut history).await;
                let r = response.unwrap();
                println!("");
                println!("{}", r);
                println!("");
            }
            cli_prompt();
        }
    }
}

fn cli_prompt() {
    print!("> ");
    stdout().flush().unwrap();
}

async fn cli_process_message(
    msg: &str,
    embeddings: &Embeddings,
    client: &Client,
    history: &mut History<'_>,
) -> Result<String> {
    let user_msg = Message::user(msg)?;

    history.user(user_msg.clone());

    let pruned_messages = history.prune_history();

    let history_size = pruned_messages.iter().map(|m| m.tokens).sum::<u16>();

    let emb = timer!("get_embedding", {
        client.get_embedding(msg).await?
    });
    let (context_msg, _context_info) = timer!("prepare_context", {
        embeddings.prepare_context(&emb, MAX_TOKENS - history_size - RESPONSE_SIZE)?
    });

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
    Ok(resp.content)
}
