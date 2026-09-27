"""python mat/check_contamination.py : share of each eval exam's 8-word shingles (sources + questions + key answers) found in the SFT data."""
import json, re, glob
grams = lambda t, n=8: (lambda w: {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)})(re.findall(r"\w+", t.lower()))
sft = set()
for f in glob.glob("data/sft/*.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); it = r["item"]
        sft |= grams(" ".join([it.get("source_text") or "", it.get("question") or "", r.get("answer") or ""]))
for x in sorted(glob.glob("data/exams/*")):
    ex = json.load(open(f"{x}/exam.json")); key = json.load(open(f"{x}/key.json"))
    g = set()
    for it in ex["items"]: g |= grams((it.get("source_text") or "") + " " + it["question"])
    for k in key: g |= grams(k.get("correct_answer") or "")
    print(f"{x.split('/')[-1]:5} {len(g & sft) / len(g):.1%} of {len(g)} shingles")
