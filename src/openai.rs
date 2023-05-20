use crate::{CHAT_MODEL, EMBEDDING_MODEL};
use anyhow::Error;
use async_openai::{
    types::{
        ChatCompletionRequestMessage, ChatCompletionResponseMessage,
        CreateChatCompletionRequestArgs, CreateEmbeddingRequestArgs,
    },
    Client as OpenAIClient,
};
use ndarray::Array1;

pub struct Client {
    client: OpenAIClient,
}

impl Client {
    pub fn new(api_key: &str) -> Self {
        let client = OpenAIClient::new().with_api_key(api_key);
        Self { client }
    }

    pub async fn get_embedding(&self, buffer: &str) -> Result<Array1<f32>, Error> {
        let request = CreateEmbeddingRequestArgs::default()
            .model(EMBEDDING_MODEL)
            .input([buffer])
            .build()?;

        let response = self.client.embeddings().create(request).await?;
        let emb = Array1::from_vec(response.data[0].embedding.clone());
        Ok(emb)
    }

    pub async fn chat(
        &self,
        messages: &[ChatCompletionRequestMessage],
    ) -> Result<ChatCompletionResponseMessage, Error> {
        let request = CreateChatCompletionRequestArgs::default()
            .model(CHAT_MODEL)
            .messages(messages)
            .build()?;

        let response = self.client.chat().create(request).await?;

        response
            .choices
            .into_iter()
            .next()
            .ok_or(anyhow::anyhow!("No reponse"))
            .map(|c| c.message)
    }
}
