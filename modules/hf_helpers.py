import requests
import logging
import os
from dotenv import load_dotenv

load_dotenv()

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
log = logging.getLogger(__name__)

def hf_infer(model: str, text: str):
    try:
        resp = requests.post(
            f"https://api-inference.huggingface.co/models/{model}",
            headers={"Authorization": f"Bearer {HF_API_TOKEN}"},
            json={"inputs": text},
            timeout=15
        )
        resp.raise_for_status()
        out = resp.json()

        # Normalize API output from dict → list[]
        if isinstance(out, dict):
            out = [out]
        elif isinstance(out, list) and len(out) == 1 and isinstance(out[0], list):
            out = out[0]

        # Filter invalid entries
        out = [x for x in out if isinstance(x, dict) and "label" in x and "score" in x]

        return out
    
    except Exception as e:
        log.warning(f"HF inference failed ({model}): {e}")
        return []
