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

# # Example: Find recent papers on a specific topic
# def find_papers_on_topic(df, topic, max_results=5):
#     """Find papers related to a specific topic."""
#     # Search in titles and abstracts
#     mask = (
#         df['title'].str.contains(topic, case=False) |
#         df['abstract'].str.contains(topic, case=False)
#     )
#     results = df[mask].sort_values('update_date', ascending=False).head(max_results)
#     return results[['title', 'authors', 'categories', 'update_date']]

# # Example: Analyze papers by year
# # Extract year from update_date
# arxiv_df['year'] = arxiv_df['update_date'].str[:4]
# papers_by_year = arxiv_df['year'].value_counts().sort_index()

# print("\nPapers published by year:")
# print(papers_by_year)

# # Example: Find papers from a specific author
# def find_papers_by_author(df, author_name, max_results=5):
#     """Find papers by a specific author."""
#     mask = df['authors'].str.contains(author_name, case=False)
#     results = df[mask].sort_values('update_date', ascending=False).head(max_results)
#     return results[['title', 'authors', 'categories', 'update_date']]

# # Example usage
# author = "Yoshua Bengio"
# author_papers = find_papers_by_author(arxiv_df, author)
# print(f"\nPapers by {author}:")
# print(author_papers)