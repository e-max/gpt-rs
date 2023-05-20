pub mod embeddings;
pub mod history;
pub mod html;
pub mod openai;
pub mod websocket;

pub const EMBEDDING_SIZE: usize = 1536;
pub const CHAT_MODEL: &'static str = "gpt-3.5-turbo";
pub const EMBEDDING_MODEL: &'static str = "text-embedding-ada-002";
pub const DATA_DIR: &'static str = "./data";
pub const HISTORY_DIR: &'static str = "./history";

pub const MAX_TOKENS: u16 = 4096;
pub const MAX_HISTORY: u16 = 1024;
pub const RESPONSE_SIZE: u16 = 512;
