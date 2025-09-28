import json
import re
import os

from .vocab import update_vocab_from_record, DEFAULT_PATH

_CODEBLOCK_RE = re.compile(
    r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE
)

def _strip_code_fences(text: str):
    """Strip markdown code fences if they exist."""

    m = _CODEBLOCK_RE.match(text.strip())
    if m:
        return m.group(1), True
    return text, False

def save_to_jsonl(text: str, path: str) -> None:
    """save AI's response to jsonl if they are legal json"""

    cleaned, _ = _strip_code_fences(text)

    try:
        obj = json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"Not legal JSON: {e}")
    
    os.makedirs(os.path.dirname(path or "."), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    update_vocab_from_record(obj, path=DEFAULT_PATH)