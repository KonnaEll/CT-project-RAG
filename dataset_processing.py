import json
from typing import List, Dict

# Convert single paper dict to source format or None if no abstract.
def transform_paper_to_source(paper: dict) -> dict:
    abstract = paper.get('abstract', '').strip()
    if not abstract:
        return None
    return {
        'text': abstract,
        'metadata': {
            'authors': paper.get('authors', []),
            'title': paper.get('title', ''),
            'update_date': paper.get('update_date', '')
        }
    }

# go through the input JSONL and write only the first `demo_count` valid sources to a new JSONL file
def create_demo_jsonl(input_path: str, demo_output_path: str, demo_count: int = 20) -> None:
    demo_written = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(demo_output_path, 'w', encoding='utf-8') as outfile_demo:

        for line in infile:
            if demo_written >= demo_count:
                break

            paper = json.loads(line)
            source = transform_paper_to_source(paper)
            if source is None:
                continue

            outfile_demo.write(json.dumps(source, ensure_ascii=False) + '\n')
            demo_written += 1

    print(f"Saved {demo_written} demo sources to {demo_output_path}")

def load_sources_jsonl(input_path: str) -> List[Dict]:
    sources: List[Dict] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            sources.append(json.loads(line))
    return sources
    
