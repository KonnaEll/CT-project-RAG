import kagglehub
import os
from datasets import load_dataset
import pandas as pd

# Step 1: Download the ArXiv dataset using kagglehub
print("Downloading ArXiv dataset from Kaggle...")
path = kagglehub.dataset_download("Cornell-University/arxiv")
print(f"Dataset downloaded to: {path}")

# List files in the downloaded directory to confirm what we have
print("\nFiles in the downloaded directory:")
for root, dirs, files in os.walk(path):
    for file in files:
        print(f" - {os.path.join(root, file)}")

json_path = os.path.join(path, "arxiv-metadata-oai-snapshot.json")
print("\nLoading dataset with the generic JSON loader...")
arxiv_dataset = load_dataset(
    "json",
    data_files={"train": json_path},
    split="train",
)
print(f"Dataset loaded successfully with {len(arxiv_dataset)} papers")

# Sample a subset for exploration (avoids memory issues)
sample_size = 1000  # Adjust based on your computer's memory
arxiv_sample = arxiv_dataset.select(range(sample_size))
arxiv_df = pd.DataFrame(arxiv_sample)

# Display dataset information
print(f"\nDataset columns: {arxiv_df.columns.tolist()}")
print(f"\nSample data (first 3 rows):")
print(arxiv_df[['abstract', 'authors', 'categories']].head(3))


# # Example: Search for papers on a specific topic
# def search_papers(df, keyword, max_results=5):
#     """Search for papers containing a keyword in title or abstract."""
#     keyword = keyword.lower()
#     mask = (
#         df['title'].str.lower().str.contains(keyword, na=False) | 
#         df['abstract'].str.lower().str.contains(keyword, na=False)
#     )
#     return df[mask].head(max_results)

# # Search example
# topic = "transformer"
# transformer_papers = search_papers(arxiv_df, topic)
# print(f"\nPapers about {topic}:")
# for _, paper in transformer_papers.iterrows():
#     print(f"Title: {paper['title']}")
#     print(f"Authors: {paper['authors']}")
#     print(f"Categories: {paper['categories']}")
#     print(f"Published: {paper.get('update_date', 'N/A')}")
#     print("-" * 80)

# # Example: Analyze publication trends over time
# if 'update_date' in arxiv_df.columns:
#     # Extract year from update_date
#     arxiv_df['year'] = arxiv_df['update_date'].str[:4]
    
#     # Count papers per year
#     year_counts = arxiv_df['year'].value_counts().sort_index()
#     print("\nPapers by year (from sample):")
#     print(year_counts)

# print("\nSuccessfully loaded and analyzed the ArXiv dataset!")