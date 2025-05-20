import time
import logging
from typing import List, Dict, Optional, Tuple, Any
import torch
from transformers import pipeline, Pipeline, AutoTokenizer
from pleias_rag_interface import RAGWithCitations

logger = logging.getLogger(__name__)

def load_qa_models(
    rag_model_name: str = "PleIAs/Pleias-RAG-350M",
    t5_model_name: str = "google/flan-t5-large",
    t5_task: str = "text2text-generation",
    device: int = 1,
    torch_dtype=torch.float16
) -> Tuple[Optional[RAGWithCitations], Optional[Pipeline]]:

    rag = None
    t5_ppl = None

    # Load RAG model
    try:
        rag = RAGWithCitations(model_path_or_name=rag_model_name)
        print("-------RAG Loaded correctly-------")
        logger.info(f"Successfully loaded RAG model '{rag_model_name}'")
    except Exception as e:
        logger.error(f"Failed to load RAG model '{rag_model_name}': {e}")

    logger.info("-----------------------------------------------------------------")

    # Load T5 pipeline (instruction-tuned + sampling)
    try:
        tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token_id})

        t5_ppl = pipeline(
            task=t5_task,
            model=t5_model_name,
            torch_dtype=torch_dtype,
            device=device,
            # enable nucleus sampling
            do_sample=False,
            num_beams=4,
            # ensure a pad token
            pad_token_id=tokenizer.eos_token_id
        )
        print("--------T5 Loaded correctly---------")
        logger.info(f"Loaded instruction-tuned T5 '{t5_model_name}' with sampling")
    except Exception as e:
        logger.error(f"Failed to load T5 pipeline '{t5_model_name}': {e}")
        t5_ppl = None

    return rag, t5_ppl

# Run only the RAG-with-citations model. Returns a dict with keys: 'response', 'time', and optionally 'error'.
def query_rag(
    query: str,
    sources: List[Dict],
    rag: RAGWithCitations
    ) -> Dict[str, Any]:

    try:
        start = time.time()
        rag_resp = rag.generate(query, sources)
        elapsed = time.time() - start
        return {
          'response': rag_resp,
          'time': elapsed
        }
    except Exception as e:
        logger.error(f"Error querying RAG model: {e}")
        return {
            'response': None,
            'time': None,
            'error': str(e)
        }

# Pure LLM call—no retrieval context.Returns: 'response': <str generated_text or None>, 'time': <float seconds> , 'error': <str if error>
SYSTEM_PROMPT = 'You are a helpful assistant'
def query_t5(
    query: str,
    t5_ppl: Pipeline,
    system_prompt: str = SYSTEM_PROMPT,
    prefix: str = "Answer the following question:"
) -> Dict[str, Any]:

    try:
        # build a single prompt that starts with the “you are helpful assistant” line
        prompt = (
            f"{system_prompt}\n\n"
            f"{prefix}\n"
            f"Question: {query}\n"
            "Answer:"
        )

        start = time.time()
        out = t5_ppl(prompt, max_length=200)     # returns [ { "generated_text": ... } ]
        elapsed = time.time() - start

        gen = out[0].get("generated_text", "").strip()
        return {'response': gen, 'time': elapsed}
    except Exception as e:
        logger.error(f"T5 generation error: {e}")
        return {'response': None, 'time': None, 'error': str(e)}
