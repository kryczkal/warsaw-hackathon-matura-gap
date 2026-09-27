"""Retrieval context for every item of an exam (CPU, offline).

python mat/context.py EXAM_DIR OUT.json [--index data/wiki_index] [--kb-k 5 --essay-kb 6 --essay-k 4]
Closed/open items: top-N entries of the fact base (data/kb) + top-k distinct Wikipedia articles,
each trimmed to the w words that overlap the query most.
Essay: pick one of the three topics (one that names its aspects first, then best retrieval score),
store it under "<id>#topic" so answer.py shows only that topic, and retrieve per aspect.
"""
import argparse, gzip, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from text import stem, q_base, trim
import kb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exam"); ap.add_argument("out")
    ap.add_argument("--index", default="data/wiki_index")
    ap.add_argument("--k", type=int, default=10); ap.add_argument("--w", type=int, default=70)
    ap.add_argument("--kb-k", type=int, default=0, help="prepend top-N fact base entries (data/kb)")
    ap.add_argument("--essay-kb", type=int, default=2, help="KB entries per essay aspect"); ap.add_argument("--essay-k", type=int, default=3); ap.add_argument("--essay-w", type=int, default=120)
    a = ap.parse_args()
    load(a.index)
    if a.kb_k:
        kb.load()
    exam = json.loads((Path(a.exam) / "exam.json").read_text(encoding="utf-8"))
    res = {}
    for it in exam["items"]:
        res.update(item_ctx(it["id"], it, a))
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", len(res))


docs = retr = None


def load(index):
    global docs, retr
    import bm25s
    R = Path(index)
    docs = [json.loads(l) for l in gzip.open(R / "docs.jsonl.gz", "rt", encoding="utf-8")]
    retr = bm25s.BM25.load(str(R / "bm25"))


def get(q, k, w, pool=5):
    ids, sc = retr.retrieve([stem(q)[:400] or ["x"]], k=k * pool, show_progress=False)
    qst = set(stem(q))
    seen, ctx = set(), []
    for di in ids[0]:
        d = docs[int(di)]
        if d["t"] in seen:
            continue
        seen.add(d["t"]); ctx.append(f"[{d['t']}] {trim(d['d'], qst, w)}")
        if len(ctx) >= k:
            break
    return ctx, float(sc[0][:k].sum())


def topics(q):
    head, _, rest = q.partition("\n1. ")
    parts = re.split(r"\n(?=[23]\. )", "1. " + rest)
    return head, [re.sub(r"\s*\n\s*", " ", p).strip() for p in parts]


def aspects(t):
    m = re.search(r"aspekt(?:y|ów)?:?\s+([^.]+?)(?:\.|$)", t)
    if not m:
        m = re.search(r"aspekt\s+([^.]+?)(?:\.|$)", t)
    if not m:
        return None
    xs = re.split(r",\s*|\s+i\s+|\s+oraz\s+", m.group(1))
    return [x.strip() for x in xs if x.strip()] if len(xs) >= 2 else None


def essay_ctx(it, a):
    """-> (chosen topic text, ctx)"""
    head, ts = topics(it["question"])
    best = None
    for n, t in enumerate(ts, 1):
        thesis = re.sub(r"^\d\.\s*", "", t).split("Zajmij stanowisko")[0]
        asp = aspects(t)
        qs = [thesis + " " + x for x in asp] if asp else [thesis]
        ctx, s = [], 0.0
        for q in qs:
            c, sc = get(q, a.essay_k, a.essay_w); ctx += kb.get(q, a.essay_kb if a.kb_k else 0) + c; s += sc / len(qs)
        cand = ((1 if asp else 0), s, n, t, ctx)
        if best is None or cand[:2] > best[:2]:
            best = cand
    if best is None:
        return None, None
    return best[3], "\n".join(dict.fromkeys(best[4]))


def item_ctx(key, it, a):
    if it.get("max_points", 0) >= 10:
        t, c = essay_ctx(it, a)
        return {key: c, key + "#topic": t} if t else {}
    c, _ = get(q_base(it), a.k, a.w)
    return {key: "\n".join(kb.get(q_base(it), a.kb_k) + c)}


if __name__ == "__main__":
    main()
