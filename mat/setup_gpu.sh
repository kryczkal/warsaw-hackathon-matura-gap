#!/usr/bin/env bash
# GPU box (CUDA, 24 GB+): build llama.cpp and the GGUFs into gguf/
#   base-Q4_K_M.gguf 7.38 GB (before) | tuned-Q5_K_M.gguf 8.55 GB (after) | base-mmproj.gguf 0.12 GB (vision)
# Needs HF access to google/gemma-4-12B (accept the license, `hf auth login`).
set -euxo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
L=${LLAMA:-$ROOT/llama.cpp}; G=${GGUF:-$ROOT/gguf}; B=${BASE:-$ROOT/models/gemma-4-12B}
mkdir -p "$G"
[ -d "$L" ] || git clone -q https://github.com/ggml-org/llama.cpp "$L"
git -C "$L" checkout -q 6f856c70990a1b5a159b0d4b2355160c83c3697b
cmake -S "$L" -B "$L/build" -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH:-native} -DLLAMA_CURL=OFF
cmake --build "$L/build" -j --target llama-server llama-quantize llama-export-lora
[ -f "$ROOT/adapter/adapter_model.safetensors" ] || curl -fL -o "$ROOT/adapter/adapter_model.safetensors" \
  https://github.com/kryczkal/warsaw-hackathon-matura-gap/releases/download/v1.0/adapter_model.safetensors
echo "66f7ac19f1d18d172fff5008d5ff5264  $ROOT/adapter/adapter_model.safetensors" | md5sum -c
pip install -q -r "$ROOT/requirements.txt" -r "$ROOT/requirements-gpu.txt"
hf download google/gemma-4-12B --local-dir "$B"
python "$L/convert_hf_to_gguf.py" "$B" --mmproj --outtype f16 --outfile "$G/base-mmproj.gguf"
python "$L/convert_hf_to_gguf.py" "$B" --outtype bf16 --outfile "$G/base-bf16.gguf"
"$L/build/bin/llama-quantize" "$G/base-bf16.gguf" "$G/base-Q4_K_M.gguf" Q4_K_M
python "$L/convert_lora_to_gguf.py" --base "$B" --outtype bf16 --outfile "$G/lora.gguf" "$ROOT/adapter"
"$L/build/bin/llama-export-lora" -m "$G/base-bf16.gguf" --lora "$G/lora.gguf" -o "$G/tuned-bf16.gguf"
"$L/build/bin/llama-quantize" "$G/tuned-bf16.gguf" "$G/tuned-Q5_K_M.gguf" Q5_K_M
rm -f "$G/tuned-bf16.gguf" "$G/base-bf16.gguf"
ls -la "$G"
