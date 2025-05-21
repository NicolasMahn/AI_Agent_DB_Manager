from chromadb.utils import embedding_functions
from scrt import OPENAI_KEY

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-ada-002",
        api_key=OPENAI_KEY
    )