# core/vocab.py
import json
import os
from typing import Any, Dict, List

DEFAULT_PATH = "data/vocab.json"
CATEGORIES = ["company", "title", "skills", "degrees", "majors"]

def _normalize(s: str) -> str:
    return (s or "").strip().lower()

def load_vocab(path: str = DEFAULT_PATH) -> Dict[str, List[str]]:
    if not os.path.exists(path):
        return {k: [] for k in CATEGORIES}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # keep only known categories; normalize + dedupe
    out: Dict[str, List[str]] = {k: [] for k in CATEGORIES}
    for k in CATEGORIES:
        seen = set()
        for item in (data.get(k) or []):
            v = _normalize(item)
            if v and v not in seen:
                seen.add(v)
                out[k].append(v)
    return out

def save_vocab(vocab: Dict[str, List[str]], path: str = DEFAULT_PATH) -> None:
    os.makedirs(os.path.dirname(path or "."), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

def update_vocab_from_record(record: Dict[str, Any], path: str = DEFAULT_PATH) -> None:
    """
    Pull terms from a single extracted record and append to vocab.json (deduped, lowercased).
    Expected record schema matches your extractor: meta/company, meta/title,
    skills.required/preferred[].name, education.degrees/majors[].
    """
    vocab = load_vocab(path)

    def add(cat: str, value: str):
        v = _normalize(value)
        if v and v not in vocab[cat]:
            vocab[cat].append(v)

    meta = record.get("meta") or {}
    if meta.get("company"): add("company", meta["company"])
    if meta.get("title"):   add("title", meta["title"])

    skills = record.get("skills") or {}
    for bucket in ("required", "preferred"):
        for it in (skills.get(bucket) or []):
            name = (it or {}).get("name")
            if name:
                add("skills", name)

    edu = record.get("education") or {}
    for d in (edu.get("degrees") or []):
        add("degrees", d)
    for m in (edu.get("majors")  or []):
        add("majors", m)

    save_vocab(vocab, path)
