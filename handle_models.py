import json
import os
import time
import pickle
import logging
from typing import List, Dict, Optional, Tuple
import torch
from transformers import pipeline, Pipeline
from pleias_rag_interface import RAGWithCitations

from typing import TypedDict, Any, Optional

logger = logging.getLogger(__name__)

class SingleResult(TypedDict, total=False):
    response: Any
    time: float
    error: str

def load_qa_models(
    rag_model_name: str = "PleIAs/Pleias-RAG-350M",
    t5_model_name: str = "google/t5-base",
    t5_task: str = "text2text-generation",
    device: int = 0,
    torch_dtype=torch.float16
) -> Tuple[Optional[RAGWithCitations], Optional[Pipeline]]:
    """
    Load both a RAG-with-citations model and a T5 generation pipeline with error handling.

    Returns:
        rag   : an instance of RAGWithCitations ready for retrieval + generation, or None on failure
        t5_ppl: a transformers pipeline for text2text-generation (T5), or None on failure
    """
    rag = None
    t5_ppl = None

    # Load RAG model
    try:
        rag = RAGWithCitations(model_path_or_name=rag_model_name)
        logger.info(f"Successfully loaded RAG model '{rag_model_name}'")
    except Exception as e:
        logger.error(f"Failed to load RAG model '{rag_model_name}': {e}")

    # Load T5 pipeline
    try:
        t5_ppl = pipeline(
            task=t5_task,
            model=t5_model_name,
            torch_dtype=torch_dtype,
            device=device
        )
        logger.info(f"Successfully initialized T5 pipeline with model '{t5_model_name}'")
    except Exception as e:
        logger.error(f"Failed to initialize T5 pipeline '{t5_model_name}': {e}")

    return rag, t5_ppl

def query_models(
    query: str,
    sources: List[Dict],
    rag: RAGWithCitations,
    t5_ppl: Pipeline
) -> Dict[str, Dict]:
    """
    Query both RAG and T5 models, record responses and timings.

    Args:
        query: the input query string
        sources: list of source dicts for RAG
        rag: initialized RAGWithCitations instance
        t5_ppl: initialized transformers pipeline for T5

    Returns:
        A dict with keys 'rag' and 't5', each mapping to a dict:
            {
                'response': model-specific response object,
                'time': elapsed seconds (float)
            }
    """
    results = {}

    # Query RAG
    try:
        start = time.time()
        rag_resp = rag.generate(query, sources)
        elapsed = time.time() - start
        results['rag'] = {
            'response': rag_resp,
            'time': elapsed
        }
    except Exception as e:
        logger.error(f"Error querying RAG model: {e}")
        results['rag'] = {
            'response': None,
            'time': None,
            'error': str(e)
        }

    # Query T5
    try:
        start = time.time()
        t5_out = t5_ppl(query)
        elapsed = time.time() - start
        results['t5'] = {
            'response': t5_out,
            'time': elapsed
        }
    except Exception as e:
        logger.error(f"Error querying T5 pipeline: {e}")
        results['t5'] = {
            'response': None,
            'time': None,
            'error': str(e)
        }

    return results

def model_check(rag,t5):

    # Quick check if both models loaded
    if rag:
        try:
            print("RAG version:", rag.__version__)
        except Exception as e:
            logger.error(f"Error accessing RAG version: {e}")
    else:
        print("RAG model is not available.")

    if t5:
        try:
            out = t5("translate English to French: The weather is nice today.")
            print("T5 test:", out[0].get("generated_text", ""))
        except Exception as e:
            logger.error(f"Error running T5 pipeline: {e}")
    else:
        print("T5 pipeline is not available.")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    rag, t5 = load_qa_models()
    model_check(rag, t5)