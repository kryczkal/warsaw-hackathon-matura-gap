# warsaw-hackathon-matura-gap

Our entry for the "max improvement" track of the Warsaw Model Trainers hackathon (Kolektyw3 × AI Tinkerers, Sep 2026): the gain of a small model on the Polish history matura (CKE, extended level, 60 points) from before to after our work.

| exam | before | after | gap |
|---|---|---|---|
| `2405` CKE May 2024 | 2 | 31 | +29 |
| `2505` CKE May 2025 | 8 | 35 | +27 |
| `2605` CKE May 2026 | 6 | 39 | +33 |
| `mock` CKE May 2023, the organizers' sample package | 7 | 39 | +32 |
| **avg** | **5.75** | **36.0** | **+30.25 (50.4 pp)** |

**Before:** the untouched `google/gemma-4-12B` base model, quantized to Q4_K_M (7.38 GB). It gets each item's own text and images, nothing else, and answers greedily.

**After:** the same model with our LoRA ([weights](https://github.com/kryczkal/warsaw-hackathon-matura-gap/releases/tag/v1.0), 500 MB) merged in, quantized to Q5_K_M (8.55 GB, plus a 0.12 GB vision projector). It runs with a harness:
- offline search over Polish Wikipedia and a hand-curated fact base (`data/kb`)
- a first pass that describes each image literally
- a 5-sample majority vote on closed questions

Claude Opus graded every answer against the official CKE key with `mat/GRADER.md`, with before and after graded side by side. Regrading the same answers moves a score by up to ±4, so read the gap, not single points.

## Reproduce

Print the table from the shipped grades:

    python mat/score.py

Regrade with [Claude Code](https://claude.com/claude-code):

    bash mat/grade.sh data/exams/2505 results/answers/2505_base.json results/answers/2505_tuned.json regrade
    python mat/score.py regrade

Regenerate the answers. You need a CUDA GPU with 24 GB and access to `google/gemma-4-12B` on Hugging Face:

    bash mat/setup_gpu.sh      # llama.cpp, LoRA weights, both GGUFs into gguf/
    bash mat/eval.sh base  data/exams/2505 new/2505_base.json
    bash mat/eval.sh tuned data/exams/2505 new/2505_tuned.json results/ctx/2505.json

Wording differs between reruns, since closed questions are sampled and llama.cpp batches requests. Scores should stay within grader noise.

On a new exam, drop the last argument and `eval.sh` retrieves the context itself. That needs the Wikipedia index, built once on CPU (about 20 min):

    pip install -r requirements.txt
    hf download wikimedia/wikipedia --repo-type dataset --include "20231101.pl/*" --local-dir data/wiki
    python mat/build_wiki.py data/wiki/20231101.pl data/wiki_index

This rebuilds the shipped contexts in `results/ctx` exactly (checked). Nothing goes online at exam time.

## Training

Untouched, `google/gemma-4-12B` scores near 0. We teach it the exam with a LoRA (`mat/train.py`):
- **Settings:** rank 32 on every attention and MLP projection of the language model, with the vision tower frozen. bf16, AdamW, cosine schedule.
- **Prompt:** each record is one exam item, in the same chat prompt `answer.py` uses at test time (`mat/common.py`). Loss is on the answer tokens only.
- **Context:** half the items carry retrieved context, so the model learns to use the materials when they help and to answer without them otherwise.
- **Stage 1:** 2 epochs, lr 1e-4, context from Wikipedia only.
- **Stage 2:** continues from stage 1 for 1 epoch at lr 5e-5, on the same items. Their context is now retrieved exactly as at test time (fact base + Wikipedia, up to 6k tokens).

The data has two sources:
- **CKE papers:** 494 items from CKE history papers 2015–2022, plus the 2023 papers for the old curriculum. Answers are written in exam style from the official keys.
- **Synthetic:** 1,465 matura-style items and essays written by Claude, spread over every period in the curriculum.

The fact base in `data/kb` (2,162 short entries: dates, people, terms, how to read maps and cartoons) was also written by Claude.

## Retrain

On one A100 this takes a few hours:

    pip install -r requirements.txt -r requirements-gpu.txt
    python mat/train.py --data data/sft/stage1.jsonl --out out/stage1 --epochs 2 --bs 2 --accum 8
    python mat/train.py --data data/sft/stage2.jsonl --out out/stage2 --epochs 1 --bs 1 --accum 16 --max-len 6144 --lr 5e-5 --init-adapter out/stage1/adapter

None of the four test exams is in the data. `python mat/check_contamination.py` finds 2–3% shared 8-word phrases, and these are citation lines and famous primary sources that CKE quotes again and again.

## Layout

    mat/eval.sh        one exam, before or after          mat/grade.sh, score.py   grading, table
    mat/run.sh         llama-server + answer.py           mat/setup_gpu.sh         builds the GGUFs
    mat/answer.py      prompts, image pass, voting        mat/train.py             LoRA training
    mat/context.py     retrieval (Wikipedia + fact base)  mat/build_wiki.py        Wikipedia index
    data/exams/        the four test exams (exam.json, key.json, images)
    data/kb/           fact base, 1 entry per line
    data/sft/          training items
    adapter/           LoRA config (weights in release v1.0)
    results/           answers, grades and retrieved contexts behind the table
