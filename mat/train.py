"""LoRA fine-tuning of the Gemma-4-12B base checkpoint on matura items (one CUDA GPU, bf16).

python mat/train.py --data data/sft/stage1.jsonl --out out/stage1 [--init-adapter out/x/adapter]
data lines: {"item": {...}, "answer": "...", "context": optional str}
"""
import argparse, json, math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mat.common import SYSTEM, user_text

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="google/gemma-4-12B")
ap.add_argument("--template-from", default="google/gemma-4-12B-it")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--bs", type=int, default=4)
ap.add_argument("--accum", type=int, default=4)
ap.add_argument("--max-len", type=int, default=3072)
ap.add_argument("--merge", action="store_true")
ap.add_argument("--init-adapter", default=None, help="continue training this LoRA adapter")
args = ap.parse_args()

import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

tok = AutoTokenizer.from_pretrained(args.model)
if not tok.chat_template:
    tok.chat_template = (Path(__file__).parent / "gemma4_it_template.jinja").read_text()
    print("chat template copied from", args.template_from)


def encode(rec):
    msgs = [{"role": "user", "content": SYSTEM + "\n\n" + user_text(rec["item"], rec.get("context"))}]
    p = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    f = p + rec["answer"].strip() + "<turn|>\n"
    pi = tok(p, add_special_tokens=False)["input_ids"]
    ci = tok(f[len(p):], add_special_tokens=False)["input_ids"]
    ids = (pi + ci)[: args.max_len]
    labels = ([-100] * len(pi) + ci)[: args.max_len]
    return ids, labels


recs = [json.loads(l) for l in open(args.data, encoding="utf-8")]
data = [encode(r) for r in recs]
print("examples", len(data), "max len", max(len(d[0]) for d in data), "tokens", sum(len(d[0]) for d in data))
print("SAMPLE:\n" + tok.decode(data[0][0])[-1500:])

model = AutoModelForImageTextToText.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
# LoRA on the language model's linear layers only (vision/audio towers untouched)
cfg = LoraConfig(r=args.rank, lora_alpha=args.rank, lora_dropout=0.05, bias="none",
                 target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
if args.init_adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    print("init from", args.init_adapter)
else:
    model = get_peft_model(model, cfg)
model.print_trainable_parameters()

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(data) / (args.bs * args.accum))
total = int(steps_per_epoch * args.epochs)
sched = get_cosine_schedule_with_warmup(opt, max(5, total // 20), total)
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
model.train()
step = 0
random.seed(0)
while step < total:
    order = sorted(range(len(data)), key=lambda i: random.random())
    for b0 in range(0, len(order), args.bs * args.accum):
        block = order[b0:b0 + args.bs * args.accum]
        ntok = sum(sum(1 for x in data[i][1] if x != -100) for i in block)
        tot_loss = 0.0
        for m0 in range(0, len(block), args.bs):
            mb = [data[i] for i in block[m0:m0 + args.bs]]
            L = max(len(x[0]) for x in mb)
            ids = torch.tensor([x[0] + [pad] * (L - len(x[0])) for x in mb], device="cuda")
            lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in mb], device="cuda")
            att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in mb], device="cuda")
            logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), lab[:, 1:].reshape(-1), ignore_index=-100, reduction="sum") / ntok
            loss.backward(); tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 5 == 0 or step == 1:
            print(f"step {step}/{total} loss {tot_loss:.4f} lr {sched.get_last_lr()[0]:.2e}", flush=True)
        if step >= total:
            break

model.save_pretrained(args.out + "/adapter")
tok.save_pretrained(args.out + "/adapter")
if args.merge:
    model = model.merge_and_unload()
    model.save_pretrained(args.out + "/merged", safe_serialization=True)
    tok.save_pretrained(args.out + "/merged")
    from transformers import AutoProcessor
    try:
        pr = AutoProcessor.from_pretrained(args.model)
        if not getattr(pr, "chat_template", None):
            pr.chat_template = tok.chat_template
        pr.save_pretrained(args.out + "/merged")
    except Exception as e:
        print("processor save failed", e)
print("DONE")
