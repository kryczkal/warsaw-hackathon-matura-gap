#!/usr/bin/env bash
# mat/run.sh MODEL.gguf EXAM_DIR OUT.json baseline|harness [answer.py args...]
# Starts llama-server on the GGUF, answers the exam with mat/answer.py, stops the server.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
M=$1; EXAM=$2; OUT=$3; MODE=$4; shift 4
LC=${LLAMA:-$ROOT/llama.cpp}/build/bin
MMPROJ=${MMPROJ:-${GGUF:-$ROOT/gguf}/base-mmproj.gguf}
PORT=${PORT:-8093}
mkdir -p "$(dirname "$OUT")" logs
# vision projector stays on CPU: on CUDA its f16 image embedding overflows and the model outputs garbage
$LC/llama-server -m "$M" --mmproj "$MMPROJ" -c ${CTX:-131072} -np ${NP:-16} -ngl 99 -b 4096 -ub 4096 --no-mmproj-offload ${SRVX:-} \
  --jinja --chat-template-file "$ROOT/mat/gemma4_it_template.jinja" --port $PORT --host 127.0.0.1 > "logs/server-$(basename "$OUT" .json).log" 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 2; done
python "$ROOT/mat/answer.py" --exam "$EXAM" --out "$OUT" --mode "$MODE" --workers ${NP:-16} --url http://127.0.0.1:$PORT/v1/chat/completions "$@"
