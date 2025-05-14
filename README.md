# CT-Project-RAG

A Retrieval-Augmented Generation (RAG) system built on ArXiv scientific papers, comparing a standard T5 model to PleIAs-RAG-350M. This repository contains scripts for data download, preprocessing, model handling, and model comparison.

---

## 📂 Repository Structure

- **download_arxiv/**  
  - Script to fetch the raw ArXiv metadata JSONL file (as provided by Konstantina).

- **dataset_processing/**  
  - Transforms the raw JSONL into the `sources` format (`abstract` + `metadata`).  
  - Includes helper functions for filtering, sampling, and saving JSONL files (full and demo subsets).

- **handle_models/**  
  - Functions to load and initialize both RAG (PleIAs-RAG-350M) and T5 models.  
  - Includes error handling, timing wrappers, and query functions.

- **model_comparison/**  
  - Jupyter notebook (or Python script) that runs interactive experiments.  
  - Allows you to input queries and gather model responses, with recorded latencies.  
  - Designed to run on GPU; Colab-friendly setup included.

- **arxiv_sources.jsonl** (4.4 GB)  
  - Full ArXiv `sources` dataset for production-scale RAG evaluation.

- **arxiv_demo_sources.jsonl** (~<100 MB)  
  - Smaller subset for quick iteration and testing; committed to Git for convenience.

---
## 🔧 Next Steps

- **Experimentation**  
  Run a diverse set of science-related questions through both models. Record:  
  - Accuracy and relevance of answers  
  - Response time (already captured)  
  - Fluency and citation quality (RAG)  
  - Any additional metrics (e.g. retrieval precision)

- **Evaluation**  
  Aggregate metrics, compare T5 vs. RAG, and document insights.

- **Optimization**  
  Ensure GPU utilization in `model_comparison`. Add multi-threaded or batch querying to speed up large-scale tests.
