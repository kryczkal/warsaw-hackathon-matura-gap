"""python mat/score.py [GRADES_DIR]  -> before/after table from {exam}_base.json / {exam}_tuned.json"""
import json, sys
from pathlib import Path
d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/grades")
rows = []
for b in sorted(d.glob("*_base.json")):
    t = d / b.name.replace("_base", "_tuned")
    if t.exists():
        gb, gt = json.load(open(b)), json.load(open(t))
        rows.append((b.name[:-10], gb["total"], gt["total"], gb["max"]))
print(f"{'exam':6} {'before':>6} {'after':>6} {'gap':>5}")
for x, a, c, m in rows:
    print(f"{x:6} {a:6} {c:6} {c - a:+5}")
n = len(rows); A = sum(r[1] for r in rows) / n; C = sum(r[2] for r in rows) / n; M = rows[0][3]
print(f"{'avg':6} {A:6.2f} {C:6.2f} {C - A:+5.2f}  ({100 * (C - A) / M:.1f} pp of {M})")
