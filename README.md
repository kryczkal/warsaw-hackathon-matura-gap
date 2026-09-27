# warsaw-hackathon-matura-gap

Our entry for the **biggest improvement** track at the Warsaw Model Trainers hackathon (Kolektyw3 × AI Tinkerers, Sep 2026).

We took `google/gemma-4-12B` and taught it the Polish history matura (CKE, extended level, 60 points). Before, it scores about 6/60. After, about 36/60. That's +30 points, or +50 percentage points.

| exam | before | after | gap |
|---|---|---|---|
| `2405` CKE May 2024 | 2 | 31 | +29 |
| `2505` CKE May 2025 | 8 | 35 | +27 |
| `2605` CKE May 2026 | 6 | 39 | +33 |
| `mock` CKE May 2023 (the organizers' sample) | 7 | 39 | +32 |
| **avg** | **5.75** | **36.0** | **+30.25 (50.4 pp)** |

**Before:** the untouched base model, quantized to Q4_K_M (7.38 GB). It gets each question's text and images. Nothing else.

**After:** the same model with our LoRA merged in, quantized to Q5_K_M (8.55 GB, plus a 0.12 GB vision part). A harness runs around it:
- It looks things up offline in Polish Wikipedia and in our own fact base.
- It first describes each image in plain words.
- On closed questions it answers 5 times and takes the majority.

## Check it fast

Everything behind the table is in this repo:
- `results/answers/`: every answer, before and after, for all four exams.
- `results/grades/`: points and a one-line reason for each item.
- `results/ctx/`: the exact text the harness retrieved for each question.
- `results/final/`: our answers on the hackathon final (`history-synthetic-c-v4`), as submitted. They come from commit `c5cd59b` (tag `v1.0`), made before our team got the questions. Later commits only touch this README and add these results.
- `data/exams/`: the four exams with their official keys.
- The LoRA weights (500 MB) are in [release v1.0](https://github.com/kryczkal/warsaw-hackathon-matura-gap/releases/tag/v1.0).

`python mat/score.py` prints the table from the grades.

Claude Opus graded every answer against the official CKE key, following `mat/GRADER.md`. It saw before and after side by side. Grading the same answers again moves a score by up to ±4, so trust the gap more than single points.

## How we trained it

One LoRA, trained in two stages (`mat/train.py`):
- **Settings:** rank 32 on every attention and MLP layer of the language model. The vision part stays frozen. bf16, AdamW, cosine schedule.
- **Prompt:** each record is one exam item, in the same chat prompt the model sees at test time (`mat/common.py`). Loss is on the answer only.
- **Stage 1:** 2 epochs at lr 1e-4. Context comes from Wikipedia only.
- **Stage 2:** 1 more epoch at lr 5e-5, starting from stage 1, on the same items. Now the context is retrieved exactly as at test time: fact base + Wikipedia, up to 6k tokens.

Half the items carry context and half don't. So the model learns to use the materials when they help, and to answer on its own when they don't.

## The data

1,959 training items, in `data/sft/`:

| source | items | what |
|---|---|---|
| CKE past papers | 494 | History papers 2015–2022, plus the 2023 papers for the old curriculum. Answers written in exam style from the official keys. |
| Synthetic | 1,465 | Matura-style questions and essays written by Claude, covering every period in the curriculum. |

976 of them come with retrieved context.

The harness searches two sources:
- **Fact base** (`data/kb/`): 2,162 short entries in 11 files, split by period, plus terms and how to read maps and cartoons. Written by Claude.
- **Polish Wikipedia:** the 2023-11-01 dump, cut into 1.29M chunks, searched with BM25. Built locally, and nothing goes online at exam time.

None of the four test exams is in the training data. `python mat/check_contamination.py` finds 2–3% shared 8-word phrases. These are citation lines and famous sources that CKE quotes again and again.

## Run it yourself

Regrade the shipped answers (needs [Claude Code](https://claude.com/claude-code)):

    bash mat/grade.sh data/exams/2505 results/answers/2505_base.json results/answers/2505_tuned.json regrade
    python mat/score.py regrade

Make new answers. You need a CUDA GPU with 24 GB and access to `google/gemma-4-12B` on Hugging Face:

    bash mat/setup_gpu.sh      # llama.cpp, LoRA weights, both GGUFs into gguf/
    bash mat/eval.sh base  data/exams/2505 new/2505_base.json
    bash mat/eval.sh tuned data/exams/2505 new/2505_tuned.json results/ctx/2505.json

The wording changes between runs, since closed questions are sampled. Scores stay within grader noise.

For a new exam, drop the last argument and `eval.sh` does the retrieval itself. That needs the Wikipedia index, built once on CPU (about 20 min). It rebuilds `results/ctx` exactly.

    pip install -r requirements.txt
    hf download wikimedia/wikipedia --repo-type dataset --include "20231101.pl/*" --local-dir data/wiki
    python mat/build_wiki.py data/wiki/20231101.pl data/wiki_index

Retrain (a few hours on one A100):

    pip install -r requirements.txt -r requirements-gpu.txt
    python mat/train.py --data data/sft/stage1.jsonl --out out/stage1 --epochs 2 --bs 2 --accum 8
    python mat/train.py --data data/sft/stage2.jsonl --out out/stage2 --epochs 1 --bs 1 --accum 16 --max-len 6144 --lr 5e-5 --init-adapter out/stage1/adapter

## Layout

    mat/eval.sh        one exam, before or after          mat/grade.sh, score.py   grading, table
    mat/run.sh         llama-server + answer.py           mat/setup_gpu.sh         builds the GGUFs
    mat/answer.py      prompts, image pass, voting        mat/train.py             LoRA training
    mat/context.py     retrieval (Wikipedia + fact base)  mat/build_wiki.py        Wikipedia index
    data/exams/        the four test exams (exam.json, key.json, images)
    data/kb/           fact base, one entry per line
    data/sft/          training items
    adapter/           LoRA config (weights in release v1.0)
    results/           answers, grades and retrieved contexts behind the table
