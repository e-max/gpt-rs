use crate::{CHAT_MODEL, DATA_DIR, EMBEDDING_SIZE};
use anyhow::Error;
use async_openai::types::{ChatCompletionRequestMessage, Role};
use ndarray::{Array, Array1, Array2, ArrayView1, Axis};
use serde::{Deserialize, Serialize};
use std::{
    borrow::Cow,
    fs::File,
    io::{BufReader, Read},
    path::Path,
};
use tiktoken_rs::async_openai::num_tokens_from_messages;

#[derive(Debug)]
pub struct Embeddings {
    filenames: Vec<String>,
    embeddings: Array2<f32>,
}

#[derive(Debug, Deserialize)]
pub struct Article {
    pub title: String,
    pub body: String,
    pub tokens: usize,
}

impl ToString for Article {
    fn to_string(&self) -> String {
        format!(r#"\n\n Article {}:\n"""\n{}\n""""#, self.title, self.body)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filename<'a> {
    pub filename: Cow<'a, str>,
    pub score: f32,
}

impl<'a> Filename<'a> {
    pub fn new(filename: &'a str, score: f32) -> Self {
        Self {
            filename: filename.into(),
            score,
        }
    }
}

impl PartialEq for Filename<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}

impl Eq for Filename<'_> {}

impl PartialOrd for Filename<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        other.score.partial_cmp(&self.score)
    }
}

impl Ord for Filename<'_> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.score.partial_cmp(&self.score).unwrap()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextInfo<'a> {
    pub filenames: Vec<Filename<'a>>,
    pub size: usize,
}

impl Embeddings {
    #[tracing::instrument]
    pub fn load<R: Read + std::fmt::Debug>(reader: R) -> Result<Self, Error> {
        let mut rdr = csv::Reader::from_reader(reader);
        let mut record = csv::ByteRecord::new();

        let mut embeddings = vec![];
        let mut filenames = vec![];
        while rdr.read_byte_record(&mut record)? {
            filenames.push(String::from_utf8(record[1].to_vec())?);

            let vec: Vec<f32> = serde_json::from_slice(&record[2]).unwrap();
            embeddings.extend_from_slice(&vec);
        }
        let len = embeddings.len() / EMBEDDING_SIZE;
        let embeddings = Array::from_shape_vec((len, EMBEDDING_SIZE), embeddings)?;
        Ok(Embeddings {
            filenames,
            embeddings,
        })
    }

    pub fn embedding(&self, index: usize) -> ArrayView1<f32> {
        self.embeddings.index_axis(Axis(0), index)
    }

    pub fn top_similar<'a>(&'a self, emb: &Array1<f32>) -> Vec<Filename<'a>> {
        let similarity = self.embeddings.dot(emb);
        let mut top: Vec<Filename> = similarity
            .into_iter()
            .enumerate()
            .map(|(idx, score)| Filename::new(&self.filenames[idx], score))
            .collect();
        top.sort_unstable();
        top
    }

    pub fn prepare_context(
        &self,
        emb: &Array1<f32>,
        token_budget: u16,
    ) -> Result<(ChatCompletionRequestMessage, ContextInfo), Error> {
        let similar = self.top_similar(emb);

        let mut message = ChatCompletionRequestMessage {
        role: Role::User,
        content: "Use the below articles about the game Vallheim to answer the subsequent question. If the answer cannot be found in the articles, write 'I could not find an answer.'".to_string(),
        name: None,
    };
        let dir = Path::new(DATA_DIR);

        let mut filenames = vec![];
        let mut size = 0;

        for filename in similar {
            let article: Article = serde_json::from_reader(BufReader::new(File::open(
                dir.join(filename.filename.as_ref()),
            )?))?;
            let mut new_message = message.clone();
            new_message.content.push_str(&article.to_string());
            let num_tokens = num_tokens_from_messages(CHAT_MODEL, &[new_message.clone()])?;
            if num_tokens < (token_budget as usize) {
                message = new_message;
                filenames.push(filename);
                size += num_tokens;
            } else {
                break;
            }
        }
        Ok((message, ContextInfo { filenames, size }))
    }
}
