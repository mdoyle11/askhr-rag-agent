import pymupdf4llm
import pymupdf
import asyncio

# md_text = pymupdf4llm.to_markdown("docs/handbook/BPA 25-26 School Year Handbook.pdf")
# print(md_text[:3000])
# print("---MIDDLE---")
# print(md_text[5000:8000])

# reader = pymupdf.open("docs/handbook/BPA 25-26 School Year Handbook.pdf")
# raw_txt = ""

# for page in reader[:3]:
#     raw_txt += page.get_text() + "\n"
# reader.close()
# print(raw_txt)

async def main():
    from askhr.ingestion.loader import load_handbook
    text = await load_handbook('docs/handbook/BPA 25-26 School Year Handbook.pdf')
    print(f'Extracted {len(text)} characters')
    print(f'First 200 chars: {text[:200]}')

asyncio.run(main())