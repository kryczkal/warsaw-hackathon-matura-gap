#!/usr/bin/env bash
# mat/eval.sh base|tuned EXAM_DIR OUT.json [CTX.json]
#   base : untouched Gemma-4-12B (Q4_K_M), the exam's own text only, greedy
#   tuned: our LoRA merged in (Q5_K_M) + harness (retrieved context, image description pass, 5-vote closed items)
#   CTX.json: retrieved context; built with mat/context.py if omitted (needs data/wiki_index, see README)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); G=${GGUF:-$ROOT/gguf}
WHO=$1; EXAM=$2; OUT=$3; C=${4:-}
if [ "$WHO" = base ]; then
  CTX=65536 NP=8 bash "$ROOT/mat/run.sh" "$G/base-Q4_K_M.gguf" "$EXAM" "$OUT" baseline
else
  [ -n "$C" ] || { C=${OUT%.json}.ctx.json; python "$ROOT/mat/context.py" "$EXAM" "$C" --index "$ROOT/data/wiki_index" --kb-k 5 --essay-kb 6 --essay-k 4; }
  CTX=73728 NP=6 SRVX="--image-min-tokens 560 --image-max-tokens 1120" \
    bash "$ROOT/mat/run.sh" "$G/tuned-Q5_K_M.gguf" "$EXAM" "$OUT" harness --context "$C" --describe
fi
