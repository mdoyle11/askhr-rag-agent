from askhr.retrieval.embeddings import embed_docs
from askhr.retrieval.pinecone_client import get_index
BATCH_SIZE = 100

async def upsert_chunks(chunks: list[dict]) -> int:
    index = get_index()
    total = 0

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        contents = [chunk['content'] for chunk in batch]
        vectors = await embed_docs(contents)

        records = []
        for chunk, vector in zip(batch, vectors):
            meta = chunk['metadata']
            records.append({
                'id': f"{meta['source']}::{meta['chunk_index']}",
                'values': vector,
                'metadata': {
                    'content': chunk['content'],
                    **meta
                }
            })
        
        index.upsert(vectors=records)
        total += len(records)
    return total

if __name__ == "__main__":
    from askhr.ingestion.loader import load_handbook
    from askhr.ingestion.chunking import chunk_text
    import asyncio

    pdf_path = "docs/handbook/BPA 25-26 School Year Handbook.pdf"
    text = asyncio.run(load_handbook(pdf_path))
    chunks = chunk_text(text, pdf_path)
    total_vectors = asyncio.run(upsert_chunks(chunks))
    print(f'Upserted {total_vectors} vectors')