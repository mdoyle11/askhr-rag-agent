import pymupdf4llm
import asyncio


async def load_handbook(doc: str) -> str:
    text = await asyncio.to_thread(pymupdf4llm.to_markdown, doc)
    return text