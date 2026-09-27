"""Curated fact base (data/kb/*.jsonl, one {"t": title, "d": text} per line) with a small in-memory BM25 index."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text import stem

docs = retr = None


def load(kbdir=Path(__file__).resolve().parent.parent / "data/kb"):
    global docs, retr
    import bm25s
    docs = [json.loads(l) for f in sorted(Path(kbdir).glob("*.jsonl")) for l in open(f, encoding="utf-8") if l.strip()]
    retr = bm25s.BM25()
    retr.index([stem(d["t"] + " " + d["t"] + " " + d["d"]) for d in docs], show_progress=False)
    return len(docs)


def get(q, k):
    if not k:
        return []
    ids, _ = retr.retrieve([stem(q)[:400] or ["x"]], k=min(k, len(docs)), show_progress=False)
    return [f"[{docs[int(i)]['t']}] {docs[int(i)]['d']}" for i in ids[0]]
