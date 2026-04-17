from askhr.ingestion.loader import load_handbook
from askhr.ingestion.chunking import chunk_text
import asyncio

pdf_path = "docs/handbook/BPA 25-26 School Year Handbook.pdf"

async def pipeline(doc: str) -> list[dict]:
    text = await load_handbook(doc)
    chunks = chunk_text(text, doc)
    return chunks

if __name__ == "__main__":
    chunks = asyncio.run(pipeline(pdf_path))
    print(f'{len(chunks)} ingested')
    print(f'Average chunk size: {sum(len(chunk['content']) for chunk in chunks) / len(chunks)}')
    print(f'\nSample chunk:')  
    print(f"  Metadata: {chunks[3]['metadata']}")
    print(f"  Content preview: {chunks[3]['content'][:200]}")
    print('\nLast Chunk:')
    print(f'  Metadata: {chunks[-1]['metadata']}')
    print(f'  Content preview: {chunks[-1]['content'][:200]}')
