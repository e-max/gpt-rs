pub mod embeddings;
pub mod history;
pub mod html;
pub mod openai;
pub mod websocket;

pub const EMBEDDING_SIZE: usize = 1536;
pub const CHAT_MODEL: &str = "gpt-3.5-turbo";
pub const EMBEDDING_MODEL: &str = "text-embedding-ada-002";
pub const DATA_DIR: &str = "./data";
pub const HISTORY_DIR: &str = "./history";

pub const MAX_TOKENS: u16 = 4096;
pub const MAX_HISTORY: u16 = 1024;
pub const RESPONSE_SIZE: u16 = 512;

#[macro_export]
macro_rules! timer {
    ($label:expr, $expr:expr) => {{
        let start = std::time::Instant::now();
        let result = $expr;
        let duration = start.elapsed();
        info!("{}: time taken: {:?}", $label, duration);
        result
    }};
}
