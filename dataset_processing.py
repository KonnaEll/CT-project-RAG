import json
import os
from typing import List, Dict


def load_arxiv_json(json_path: str) -> List[Dict]:
    """
    Load the ArXiv metadata JSON lines file into a list of dicts.
    """
    papers = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            papers.append(json.loads(line))
    return papers


def transform_papers_to_sources(papers: List[Dict]) -> List[Dict]:
    """
    Transform raw ArXiv paper dicts into 'sources' format.

    Each source has:
      - text: the abstract
      - metadata: dict with authors, title, update_date
    """
    sources = []
    for paper in papers:
        abstract = paper.get('abstract', '').strip()
        if not abstract:
            continue  # skip entries without abstract

        metadata = {
            'authors': paper.get('authors', []),
            'title': paper.get('title', ''),
            'update_date': paper.get('update_date', '')
        }

        sources.append({
            'text': abstract,
            'metadata': metadata
        })
    return sources

def save_sources(sources: List[Dict], output_path: str) -> None:
    """
    Save the sources list to a JSON lines file for easy streaming.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for source in sources:
            f.write(json.dumps(source, ensure_ascii=False) + '\n')
    print(f"Saved {len(sources)} sources to {output_path}")

def save_demo_sources(sources: List[Dict], output_path: str) -> None:
    """
    Save only the first 100 sources to a JSON lines file for quick exploration.
    """
    demo_count = min(100, len(sources))
    with open(output_path, 'w', encoding='utf-8') as f:
        for source in sources[:demo_count]:
            f.write(json.dumps(source, ensure_ascii=False) + '\n')
    print(f"Saved {demo_count} demo sources to {output_path}")

def load_sources_jsonl(input_path: str) -> List[Dict]:
    """
    Read a JSON lines file into a list of sources.

    Returns:
        List of dicts with 'abstract' and 'metadata'.
    """
    sources: List[Dict] = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            sources.append(json.loads(line))
    return sources

if __name__ == '__main__':
    # Path to the downloaded ArXiv metadata JSONL file
    input_path = os.path.expanduser(
        '~/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/232/arxiv-metadata-oai-snapshot.json'
    )
    output_path = 'arxiv_sources.jsonl'
    output_path_demo = 'arxiv_demo_sources.jsonl'

    # Load, transform, and save
    print("Loading papers...")
    papers = load_arxiv_json(input_path)
    print(f"Loaded {len(papers)} papers")

    print("Transforming to sources format...")
    sources = transform_papers_to_sources(papers)

    save_sources(sources, output_path)
    save_demo_sources(sources, output_path_demo)