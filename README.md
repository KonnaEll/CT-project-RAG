# CT-Project-RAG

A Retrieval-Augmented Generation (RAG) system built on ArXiv scientific papers, comparing a standard T5 model to PleIAs-RAG-350M. This repository contains scripts for data download, preprocessing, model handling, and model comparison.

---

## 📂 Repository Structure

**dataset_processing/**  
  - Transforms the raw JSONL into the `sources` format (`abstract` + `metadata`).  
  - Includes helper functions for filtering, sampling, and saving JSONL files.

**handle_models/**  
  - Functions to load and initialize both RAG (PleIAs-RAG-350M) and T5 models.  
  - Includes error handling, timing wrappers, and query functions.

**model_comparison/**  
  - Jupyter notebook that runs interactive experiments.  
  - Allows you to input queries and gather model responses, with recorded latencies.  
  - Designed to run on CPU and GPU; Colab-friendly setup included.

**arxiv_21_sources.jsonl** (~<100 MB)  
  - Smaller subset for quick iteration and testing; committed to Git for convenience. contains 20 sources from ArXiv and a manual source for the trivial question tested

---
## 🔧 Steps

- **Experimentation**  
  Run a diverse set of science-related questions through both models. Record:  
  - Accuracy and relevance of answers  
  - Response time
  - Fluency and completeness (RAG)  

- **Evaluation**  
  Aggregate metrics, compare T5 vs. RAG, and document insights.

- **Optimization**  
  Ensure GPU utilization in `model_comparison`. Add multi-threaded or batch querying to speed up large-scale tests.
