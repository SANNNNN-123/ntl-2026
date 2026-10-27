"""Stage-1 smoke of the Run E LoRA24 recipe on pool_v1.

Two prompts that have all 18 models (36 passages). One epoch of those
prompts is exactly two optimizer steps at accum 6. The checkpoint is
written to the Modal volume `ntl-smoke-out`, not to the Kaggle `lora_out`
from Run E.

    modal run modal/smoke_pool_v1.py

Run this from the repo root. After it prints a finite loss and
COMPLETE.json, the same script is the place to point at pool_v5 / pool_v6.
Those files call the prompt column `grp` and `prompt_group`; this loader
expects `prompt_id`, which pool_v1 already has.
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
POOL_LOCAL = REPO / "data" / "pool_v1.parquet"
MODEL_ID = "Qwen/Qwen3-8B-Base"

LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]
KEEP = 24
MAX_LEN = 447
ACCUM = 6
SEED = 7
R, ALPHA, LDROP = 16, 32, 0.05
LR_ADAPTER, LR_HEAD = 1e-4, 1e-3
WD, WARMUP = 0.01, 0.10
SMOKE_PROMPTS = 2
SMOKE_STEPS = 2

app = modal.App("ntl-smoke-pool-v1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install(
        "transformers>=4.51.0",
        "peft>=0.14.0",
        "accelerate>=1.0.0",
        "pandas",
        "pyarrow",
        "safetensors",
        "numpy",
        "sentencepiece",
        "protobuf",
    )
    .env(
        {
            "HF_HOME": "/cache",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    .add_local_file(POOL_LOCAL, remote_path="/data/pool_v1.parquet")
)

cache_vol = modal.Volume.from_name("ntl-model-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ntl-smoke-out", create_if_missing=True)


def load_smoke_frame(path: str, n_prompts: int = SMOKE_PROMPTS):
    """Two full 18-model prompts. That is 36 rows, 12 triplets, 2 steps."""
    import pandas as pd

    pool = pd.read_parquet(path)
    if "prompt_id" not in pool.columns:
        raise SystemExit(
            f"{path} has columns {list(pool.columns)}; rename the prompt column to prompt_id"
        )
    n_per = pool.groupby("prompt_id").model.nunique()
    full = n_per[n_per == 18].index.tolist()
    if len(full) < n_prompts:
        full = n_per[n_per >= 3].sort_values(ascending=False).index.tolist()
    keep = full[:n_prompts]
    df = (
        pool[pool.prompt_id.isin(keep)][["id", "prompt_id", "text", "model"]]
        .dropna()
        .copy()
    )
    df["text"] = df["text"].astype(str)
    return df.reset_index(drop=True)


def triplets(rows, pid, rng, accum):
    out = []
    for p in set(pid[rows].tolist()):
        m = rng.permutation(rows[pid[rows] == p])
        if len(m) < 3:
            continue
        for s in range(0, len(m), 3):
            b = list(m[s : s + 3])
            while len(b) < 3:
                b.append(int(rng.choice([x for x in m if x not in b])))
            out.append(b)
    if not out:
        raise RuntimeError("no triplets — need >=3 answers per prompt")
    out = [out[i] for i in rng.permutation(len(out))]
    while len(out) % accum:
        out.append(out[rng.randint(len(out))])
    return out


@app.function(
    image=image,
    gpu="H100",
    timeout=45 * 60,
    volumes={"/cache": cache_vol, "/out": out_vol},
)
def smoke() -> dict:
    import importlib.util as iu
    import json
    import math
    import os
    import time

    import numpy as np
    import torch
    from safetensors.torch import save_file
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    # Kaggle's peft build rejects torchao. A clean image usually has no torchao,
    # and this keeps the import path the same as the notebook if one is present.
    find = iu.find_spec
    iu.find_spec = lambda name, *a, **k: None if name.split(".")[0] == "torchao" else find(name, *a, **k)
    from peft import LoraConfig, get_peft_model
    import peft.import_utils as pi

    pi.is_torchao_available = lambda *a, **k: False

    t0 = time.time()
    df = load_smoke_frame("/data/pool_v1.parquet")
    labels = sorted(df.model.unique())
    label_to_i = {lab: i for i, lab in enumerate(labels)}
    y = df.model.map(label_to_i).to_numpy()
    pid = df.prompt_id.to_numpy()
    print(
        f"smoke rows {len(df)} prompts {df.prompt_id.nunique()} labels {len(labels)} gpu {torch.cuda.get_device_name(0)}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tokens = [
        ids + [tok.eos_token_id]
        for ids in tok(
            df.text.tolist(),
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LEN,
        )["input_ids"]
    ]

    counts = df.model.value_counts().reindex(labels).fillna(1).to_numpy(dtype=np.float64)
    ce_weight = torch.tensor(counts.sum() / (len(labels) * counts), dtype=torch.float32, device="cuda")

    mc = AutoConfig.from_pretrained(MODEL_ID)
    hidden = int(mc.hidden_size)
    if hidden != 4096 or int(mc.num_hidden_layers) < KEEP:
        raise RuntimeError(f"unexpected base config hidden={hidden} layers={mc.num_hidden_layers}")
    mc.num_hidden_layers, mc.use_cache = KEEP, False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[:KEEP]

    dev = torch.device("cuda:0")
    base = AutoModel.from_pretrained(MODEL_ID, config=mc, dtype=torch.float16).to(dev)
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    model = get_peft_model(
        base,
        LoraConfig(
            r=R,
            lora_alpha=ALPHA,
            lora_dropout=LDROP,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()

    norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in LAYERS]).to(dev).float()
    plain = nn.Linear(hidden * len(LAYERS), len(labels)).to(dev).float()
    centered = nn.Linear(hidden * len(LAYERS), len(labels)).to(dev).float()
    drop = nn.Dropout(0.1)
    adapter_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
    opt = torch.optim.AdamW(
        [{"params": adapter_params, "lr": LR_ADAPTER}, {"params": head_params, "lr": LR_HEAD}],
        weight_decay=WD,
    )
    rng = np.random.RandomState(SEED + 99)
    plan = triplets(np.arange(len(df)), pid, rng, ACCUM)
    total = max(1, len(plan) // ACCUM)
    warm = max(1, int(WARMUP * total))
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))),
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
    opt.zero_grad(set_to_none=True)

    def rep(idx):
        batch = tok.pad({"input_ids": [tokens[i] for i in idx]}, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            hs = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        m = batch["attention_mask"].unsqueeze(-1).float()
        m[:, 0] = 0
        with torch.autocast("cuda", enabled=False):
            return torch.cat(
                [nm((hs[layer].float() * m).sum(1) / m.sum(1)) for layer, nm in zip(LAYERS, norms)],
                -1,
            )

    model.train()
    norms.train()
    step = 0
    skipped = 0
    losses = []
    for i, batch_idx in enumerate(plan):
        z = rep(batch_idx)
        yy = torch.as_tensor(y[batch_idx], device=dev)
        loss = nn.functional.cross_entropy(plain(drop(z)), yy, weight=ce_weight) + nn.functional.cross_entropy(
            centered(drop(z - z.mean(0, keepdim=True))), yy, weight=ce_weight
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at triplet {i}")
        scaler.scale(loss / ACCUM).backward()
        losses.append(float(loss.detach()))
        if (i + 1) % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(adapter_params + head_params, 1.0)
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < before:
                skipped += 1
            opt.zero_grad(set_to_none=True)
            sch.step()
            step += 1
            print(f"step {step}/{SMOKE_STEPS} loss {np.mean(losses[-ACCUM:]):.4f} skipped {skipped}", flush=True)
            if step >= SMOKE_STEPS:
                break

    outdir = "/out/smoke_pool_v1/job_stage1"
    os.makedirs(os.path.join(outdir, "adapter"), exist_ok=True)
    model.save_pretrained(os.path.join(outdir, "adapter"), safe_serialization=True)
    save_file(
        {f"norms.{k}": v.detach().cpu().contiguous() for k, v in norms.state_dict().items()},
        os.path.join(outdir, "norms.safetensors"),
    )
    tok.save_pretrained(os.path.join(outdir, "tokenizer"))
    np.savez_compressed(
        os.path.join(outdir, "readouts.npz"),
        classes=np.array(labels),
        pool_layers=np.array(LAYERS),
        plain_coef=plain.weight.detach().cpu().numpy(),
        plain_intercept=plain.bias.detach().cpu().numpy(),
        center_coef=centered.weight.detach().cpu().numpy(),
        center_intercept=centered.bias.detach().cpu().numpy(),
    )
    summary = {
        "job": "stage1",
        "smoke": True,
        "pool": "pool_v1",
        "rows": int(len(df)),
        "prompts": int(df.prompt_id.nunique()),
        "labels": labels,
        "steps": step,
        "skipped": skipped,
        "loss": float(np.mean(losses)) if losses else None,
        "secs": time.time() - t0,
        "model": MODEL_ID,
    }
    with open(os.path.join(outdir, "COMPLETE.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    out_vol.commit()
    print(f"saved {outdir} in {summary['secs']:.0f}s", flush=True)
    return summary


@app.local_entrypoint()
def main():
    if not POOL_LOCAL.is_file():
        raise SystemExit(f"missing {POOL_LOCAL}")
    result = smoke.remote()
    print(result)
