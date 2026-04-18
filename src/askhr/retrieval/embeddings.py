from langchain_google_genai import GoogleGenerativeAIEmbeddings
from askhr.config import get_settings

settings = get_settings()
_doc_embeddings = GoogleGenerativeAIEmbeddings(
    model='gemini-embedding-001',
    google_api_key=settings.google_api_key.get_secret_value(),
    output_dimensionality=settings.embedding_dimension,
    task_type='RETRIEVAL_DOCUMENT'
)

_query_embeddings = GoogleGenerativeAIEmbeddings(
    model='gemini-embedding-001',
    google_api_key=settings.google_api_key.get_secret_value(),
    output_dimensionality=settings.embedding_dimension,
    task_type='RETRIEVAL_QUERY'
)

async def embed_docs(texts: list[str]) -> list[list[float]]:
    return await _doc_embeddings.aembed_documents(texts)


async def embed_query(text: str) -> list[float]:
    return await _query_embeddings.aembed_query(text)