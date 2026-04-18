from askhr.retrieval.embeddings import embed_query
from askhr.retrieval.pinecone_client import get_index

async def search(query: str, top_k: int) -> list[dict]:
    index = get_index()
    query_vector = await embed_query(query)

    response = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True
    )

    return [
        {
         'score': match['score'],
         'content': match['metadata'].get('content', ''),
         'source': match['metadata'].get('source'),
         'section': match['metadata'].get('section'),
         'chunk_index': match['metadata'].get('chunk_index')
         }

         for match in response['matches']
    ]