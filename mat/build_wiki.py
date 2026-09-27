"""Offline Polish Wikipedia retrieval corpus (history + art/culture/economy/press) and its BM25 index.
hf download wikimedia/wikipedia --repo-type dataset --include "20231101.pl/*" --local-dir data/wiki
python mat/build_wiki.py data/wiki/20231101.pl data/wiki_index      (CPU, ~20 min) -> docs.jsonl.gz + bm25/
"""
import glob, gzip, json, os, re, sys
import pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from text import HIST, YEAR, chunks, stem

EXTRA = re.compile(r"(styl|architekt|malar|sztuk|rzeźb|kości|katedr|zamek|pałac|klasztor|gazet|czasopism|pismo|przemysł|gospodar|"
                   r"plebiscyt|parti|reform|wynalaz|uczon|filozof|pisarz|poet|literat|kultur|epok|renesans|barok|gotyk|romańsk|klasycyzm|"
                   r"romantyzm|pozytywizm|moderni|secesj|awangard|realizm|impresjonizm|oświeceni|humanizm|traktat|ruch |organizac|"
                   r"minist|premier|prezydent|generał|dowódc|stolic|miasto|prowincj|cywilizac|religi|kościół|teolog|herb|order|medal|moneta)", re.I)
src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
n = 0
with gzip.open(out + "/docs.jsonl.gz", "wt", encoding="utf-8") as f:
    for p in sorted(glob.glob(src + "/*.parquet")):
        t = pq.read_table(p, columns=["title", "text"])
        for title, text in zip(t.column("title").to_pylist(), t.column("text").to_pylist()):
            head = text[:3000]
            if not YEAR.search(head) or len(text) < 400:
                continue
            h = len(HIST.findall(head)); e = len(EXTRA.findall(head))
            if h >= 2 or (h + e >= 3):
                for c in chunks(text):
                    f.write(json.dumps({"t": title, "d": c}, ensure_ascii=False) + "\n"); n += 1
        print(p, n, flush=True)
print("chunks", n)
import bm25s
retr = bm25s.BM25()
retr.index([stem(json.loads(l)["d"]) for l in gzip.open(out + "/docs.jsonl.gz", "rt", encoding="utf-8")], show_progress=False)
retr.save(out + "/bm25")
print("indexed")
