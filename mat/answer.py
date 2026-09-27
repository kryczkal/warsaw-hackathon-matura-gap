"""Answer an exam with a running llama-server (see mat/serve.sh).

python mat/answer.py --exam EXAM_DIR --out answers.json --mode baseline|harness [--context ctx.json] [--describe]
baseline: the exam's own text, greedy, one answer per item.
harness:  our prompt + retrieved context (from mat/context.py); with --describe the model first writes a literal
          description of each item's images, which is added to the context; closed items take a 5-sample majority vote.
"""
import argparse, base64, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mat.common import SYSTEM, user_text, baseline_text, kind, vote_closed, single_topic

ap = argparse.ArgumentParser()
ap.add_argument("--exam", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mode", default="harness", choices=["baseline", "harness"])
ap.add_argument("--context", default=None)
ap.add_argument("--describe", action="store_true")
ap.add_argument("--votes", type=int, default=5)
ap.add_argument("--url", default="http://127.0.0.1:8093/v1/chat/completions")
ap.add_argument("--workers", type=int, default=8)
args = ap.parse_args()

exam_dir = Path(args.exam)
exam = json.loads((exam_dir / "exam.json").read_text(encoding="utf-8"))
items = exam["items"]
rag = json.loads(Path(args.context).read_text(encoding="utf-8")) if args.context else {}


def img_parts(item):
    out = []
    for im in item.get("images", []):
        b = base64.b64encode((exam_dir / im["path"]).read_bytes()).decode()
        out.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}})
    return out


DESCRIBE = ("Opisz dokładnie i dosłownie, co widać na ilustracji (ilustracjach) do tego zadania: przepisz wszystkie widoczne napisy, "
            "podpisy, daty i liczby; wymień postacie, stroje, symbole, herby, obiekty, budowle i ich cechy, a na mapie nazwy, granice i oznaczenia z legendy. "
            "Nie interpretuj i nie zgaduj — jeśli czegoś nie widać wyraźnie, nie wymieniaj tego.")
desc = {}


def build(item):
    if args.mode == "baseline":
        return [{"role": "user", "content": img_parts(item) + [{"type": "text", "text": baseline_text(item)}]}]
    ctx = rag.get(item["id"])
    if desc.get(item["id"]):
        ctx = "Opis ilustracji (sporządzony wcześniej, dosłowny):\n" + desc[item["id"]] + ("\n\n" + ctx if ctx else "")
    if rag.get(item["id"] + "#topic"):  # context.py picked the essay topic: keep only that one
        item = single_topic(item, rag[item["id"] + "#topic"])
    return [{"role": "user", "content": img_parts(item) + [{"type": "text", "text": SYSTEM + "\n\n" + user_text(item, ctx)}]}]


def call(msgs, seed, **sp):
    body = json.dumps({"messages": msgs, "seed": seed, **sp}).encode()
    req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())["choices"][0]["message"]["content"] or ""
        except Exception as e:
            err = e
            time.sleep(5)
    return f"[error {err}]"


if args.describe and args.mode != "baseline":
    di = [it for it in items if it.get("images")]
    dm = lambda it: [{"role": "user", "content": img_parts(it) + [{"type": "text", "text": "Zadanie z matury z historii:\n" + it["question"].strip() + "\n\n" + DESCRIBE}]}]
    with ThreadPoolExecutor(args.workers) as ex:
        for it, o in zip(di, ex.map(lambda it: call(dm(it), 0, temperature=0.0, max_tokens=500, repeat_penalty=1.05), di)):
            desc[it["id"]] = o.strip()
    Path(args.out + ".desc.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1), encoding="utf-8")

jobs = []  # (item_index, seed, msgs, sampling)
for i, it in enumerate(items):
    k = kind(it)
    msgs = build(it)
    if args.mode == "baseline":
        jobs.append((i, 0, msgs, dict(temperature=0.0, max_tokens=2048)))
    elif k == "closed":
        for v in range(args.votes):
            jobs.append((i, v, msgs, dict(temperature=0.7, top_p=0.95, max_tokens=256)))
    elif k == "essay":
        jobs.append((i, 0, msgs, dict(temperature=0.3, max_tokens=3000, repeat_penalty=1.05)))
    else:
        jobs.append((i, 0, msgs, dict(temperature=0.0, max_tokens=600, repeat_penalty=1.03)))

t = time.time()
with ThreadPoolExecutor(args.workers) as ex:
    outs = list(ex.map(lambda j: call(j[2], j[1], **j[3]), jobs))
per = {}
for (i, _, _, _), o in zip(jobs, outs):
    per.setdefault(i, []).append(o.strip())
answers, raw = [], {}
for i, it in enumerate(items):
    texts = per[i]
    final = texts[0] if len(texts) == 1 else vote_closed(texts)
    raw[it["id"]] = texts
    answers.append({"id": it["id"], "answer": final})
print(f"generated {len(answers)} answers in {time.time()-t:.0f}s")
Path(args.out).write_text(json.dumps({"exam_id": exam["exam_id"], "answers": answers}, ensure_ascii=False, indent=2), encoding="utf-8")
Path(args.out + ".raw.json").write_text(json.dumps(raw, ensure_ascii=False, indent=1), encoding="utf-8")
