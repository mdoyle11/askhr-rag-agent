from langchain_text_splitters import RecursiveCharacterTextSplitter
import re

def chunk_text(text: str, source: str) -> list[dict]:
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        chunk_size = 1000,
        chunk_overlap = 150
    )
    sections = extract_sections(text)
    chunks = []
    chunk_index = 0
    for section in sections:        
        splits = text_splitter.split_text(section['content'])
        for split in splits:
            chunk = {"content": split, "metadata": {'source': source, 'chunk_index': chunk_index, 'section': section['header']}}
            chunks.append(chunk)
            chunk_index += 1
    return chunks

def extract_sections(text: str) -> list[dict]:
    sections = []
    current = {'header': 'Introduction', 'lines': []}
    for line in text.split('\n'):
        if re.match(r'^#{1,4}\s+', line):
            if current['lines']:
                sections.append({'header': current['header'], 'content': '\n'.join(current['lines'])})
            current = {'header': line.strip('# ').strip(), 'lines': []}
        current['lines'].append(line)
    if current['lines']:
        sections.append({'header': current['header'], 'content': '\n'.join(current['lines'])})

    return sections