"""Run E two-stage LoRA24 on pool_v5, one H100.

Stage 1: 1 epoch on prompts with at least 3 models (114,801 rows).
Stage 2: 2 epochs on host train.csv, folds 0-4 then final, warm-started
from stage 1. Checkpoints go to the Modal volume `ntl-pool-v5-out`.
A finished job is skipped on a rerun.

    modal run modal/train_pool_v5.py

`grp` in pool_v5 is renamed to `prompt_id` before training.
"""

from pathlib import Path

import modal

REPO = Path(__file__).resolve().parents[1]
POOL_LOCAL = REPO / "data" / "pool_v5_122312.parquet"
HOST_LOCAL = REPO / "data" / "train.csv"
MODEL_ID = "Qwen/Qwen3-8B-Base"

LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]
KEEP = 24
MAX_LEN = 447
ACCUM = 6
SEED = 7
FOLD_SEED = 0
R, ALPHA, LDROP = 16, 32, 0.05
WD, WARMUP = 0.01, 0.10
S1_EPOCHS, S1_LR_ADAPTER, S1_LR_HEAD = 1, 1e-4, 1e-3
S2_EPOCHS, S2_LR_ADAPTER, S2_LR_HEAD = 2, 5e-5, 5e-4
S2_JOBS = ["0", "1", "2", "3", "4", "final"]

app = modal.App("ntl-train-pool-v5")

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
    .add_local_file(POOL_LOCAL, remote_path="/data/pool_v5.parquet")
    .add_local_file(HOST_LOCAL, remote_path="/data/train.csv")
)

cache_vol = modal.Volume.from_name("ntl-model-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ntl-pool-v5-out", create_if_missing=True)


def load_frame(path: str, min_models: int = 0):
    import pandas as pd

    frame = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    if "prompt_id" not in frame.columns and "grp" in frame.columns:
        frame = frame.rename(columns={"grp": "prompt_id"})
    need = {"id", "prompt_id", "text", "model"}
    missing = need - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing {sorted(missing)}")
    frame = frame[list(need)].dropna().copy()
    frame["text"] = frame["text"].astype(str)
    if min_models:
        n_per = frame.groupby("prompt_id").model.nunique()
        keep = set(n_per[n_per >= min_models].index)
        frame = frame[frame.prompt_id.isin(keep)].copy()
    return frame.reset_index(drop=True)


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
        raise RuntimeError("no triplets — need >=3 answers per prompt in the fit set")
    out = [out[i] for i in rng.permutation(len(out))]
    while len(out) % accum:
        out.append(out[rng.randint(len(out))])
    return out


@app.function(
    image=image,
    gpu="H100",
    timeout=10 * 60 * 60,
    volumes={"/cache": cache_vol, "/out": out_vol},
)
def train() -> list:
    import importlib.util as iu
    import json
    import math
    import os
    import time

    import numpy as np
    import torch
    from safetensors.torch import load_file, save_file
    from torch import nn
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    find = iu.find_spec
    iu.find_spec = lambda name, *a, **k: None if name.split(".")[0] == "torchao" else find(name, *a, **k)
    from peft import LoraConfig, PeftModel, get_peft_model
    import peft.import_utils as pi

    pi.is_torchao_available = lambda *a, **k: False

    out_root = "/out/pool_v5"
    os.makedirs(out_root, exist_ok=True)

    def run_job(df, job, epochs, lr_adapter, lr_head, class_weights, init_adapter):
        outdir = os.path.join(out_root, f"job_{job}")
        done = os.path.join(outdir, "COMPLETE.json")
        if os.path.isfile(done):
            summary = json.load(open(done, encoding="utf-8"))
            print(f"job {job} already complete, skipping", flush=True)
            return summary

        t0 = time.time()
        os.makedirs(outdir, exist_ok=True)
        labels = sorted(df.model.unique())
        if len(labels) != 18:
            raise RuntimeError(f"job {job}: expected 18 labels, got {labels}")
        label_to_i = {lab: i for i, lab in enumerate(labels)}
        y = df.model.map(label_to_i).to_numpy()
        pid = df.prompt_id.to_numpy()
        prompts = np.array(sorted(df.prompt_id.unique()))
        rng0 = np.random.RandomState(FOLD_SEED)
        rng0.shuffle(prompts)
        fold = df.prompt_id.map({p: i % 5 for i, p in enumerate(prompts)}).to_numpy()

        if job in ("final", "stage1"):
            fit = np.arange(len(df))
            valid = np.arange(min(3, len(df)))
        else:
            fit = np.where(fold != int(job))[0]
            valid = np.where(fold == int(job))[0]

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
        ce_weight = None
        if class_weights:
            counts = df.model.value_counts().reindex(labels).fillna(1).to_numpy(dtype=np.float64)
            ce_weight = torch.tensor(
                counts.sum() / (len(labels) * counts), dtype=torch.float32, device="cuda"
            )

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
        lora_cfg = LoraConfig(
            r=R,
            lora_alpha=ALPHA,
            lora_dropout=LDROP,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        if init_adapter:
            model = PeftModel.from_pretrained(base, init_adapter, is_trainable=True)
            print(f"job {job}: loaded {init_adapter}", flush=True)
        else:
            model = get_peft_model(base, lora_cfg)
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()

        norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in LAYERS]).to(dev).float()
        plain = nn.Linear(hidden * len(LAYERS), len(labels)).to(dev).float()
        centered = nn.Linear(hidden * len(LAYERS), len(labels)).to(dev).float()
        drop = nn.Dropout(0.1)
        if init_adapter:
            stage_dir = os.path.dirname(init_adapter.rstrip("/\\"))
            nsd = load_file(os.path.join(stage_dir, "norms.safetensors"))
            norms.load_state_dict({k.split("norms.", 1)[-1]: v for k, v in nsd.items()})
            with np.load(os.path.join(stage_dir, "readouts.npz"), allow_pickle=False) as z:
                classes = [str(x) for x in z["classes"].tolist()]
                if classes != labels:
                    raise RuntimeError(f"label order mismatch {classes} vs {labels}")
                with torch.no_grad():
                    plain.weight.copy_(torch.tensor(z["plain_coef"]))
                    plain.bias.copy_(torch.tensor(z["plain_intercept"]))
                    centered.weight.copy_(torch.tensor(z["center_coef"]))
                    centered.bias.copy_(torch.tensor(z["center_intercept"]))

        adapter_params = [p for p in model.parameters() if p.requires_grad]
        head_params = list(norms.parameters()) + list(plain.parameters()) + list(centered.parameters())
        opt = torch.optim.AdamW(
            [{"params": adapter_params, "lr": lr_adapter}, {"params": head_params, "lr": lr_head}],
            weight_decay=WD,
        )
        rng = np.random.RandomState(SEED + (99 if job in ("final", "stage1") else int(job)))
        plan = [triplets(fit, pid, rng, ACCUM) for _ in range(epochs)]
        total = sum(len(epoch) for epoch in plan) // ACCUM
        warm = max(1, int(WARMUP * total))
        sch = torch.optim.lr_scheduler.LambdaLR(
            opt,
            lambda s: (s + 1) / warm
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))),
        )
        scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
        opt.zero_grad(set_to_none=True)
        print(
            f"job {job} rows {len(df)} fit {len(fit)} valid {len(valid)} steps {total} warmup {warm}",
            flush=True,
        )

        def rep(idx):
            batch = tok.pad({"input_ids": [tokens[i] for i in idx]}, return_tensors="pt").to(dev)
            with torch.autocast("cuda", dtype=torch.float16):
                hs = model(**batch, output_hidden_states=True, use_cache=False).hidden_states
            mask = batch["attention_mask"].unsqueeze(-1).float()
            mask[:, 0] = 0
            with torch.autocast("cuda", enabled=False):
                return torch.cat(
                    [nm((hs[layer].float() * mask).sum(1) / mask.sum(1)) for layer, nm in zip(LAYERS, norms)],
                    -1,
                )

        model.train()
        norms.train()
        step = 0
        skipped = 0
        losses = []
        for ep, batches in enumerate(plan):
            for i, batch_idx in enumerate(batches):
                z = rep(batch_idx)
                yy = torch.as_tensor(y[batch_idx], device=dev)
                loss = nn.functional.cross_entropy(plain(drop(z)), yy, weight=ce_weight)
                loss = loss + nn.functional.cross_entropy(
                    centered(drop(z - z.mean(0, keepdim=True))), yy, weight=ce_weight
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"job {job} non-finite loss at epoch {ep} triplet {i}")
                scaler.scale(loss / ACCUM).backward()
                losses.append(float(loss.detach()))
                if (i + 1) % ACCUM != 0:
                    continue
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
                if step % 30 == 0 or step == total:
                    print(
                        f"job {job} ep {ep} step {step}/{total} loss {np.mean(losses[-180:]):.4f} "
                        f"skipped {skipped} ({time.time() - t0:.0f}s)",
                        flush=True,
                    )

        model.eval()
        norms.eval()
        features = np.zeros((len(valid), hidden * len(LAYERS)), np.float32)
        order = np.argsort([len(tokens[i]) for i in valid], kind="stable")
        with torch.inference_mode():
            for s in range(0, len(order), 6):
                pos = order[s : s + 6]
                features[pos] = rep(valid[pos]).cpu().numpy()
        if not np.isfinite(features).all():
            raise RuntimeError(f"job {job} produced non-finite features")

        plain_w = plain.weight.detach().cpu().numpy()
        plain_b = plain.bias.detach().cpu().numpy()
        center_w = centered.weight.detach().cpu().numpy()
        center_b = centered.bias.detach().cpu().numpy()
        acc = float(((features @ plain_w.T + plain_b).argmax(1) == y[valid]).mean())
        np.savez_compressed(
            os.path.join(outdir, "readouts.npz"),
            classes=np.array(labels),
            pool_layers=np.array(LAYERS),
            plain_coef=plain_w,
            plain_intercept=plain_b,
            center_coef=center_w,
            center_intercept=center_b,
        )
        np.savez_compressed(
            os.path.join(outdir, "valid_features.npz"),
            ids=df.id.to_numpy()[valid],
            features=features.astype(np.float16),
        )
        if job in ("stage1", "final"):
            model.save_pretrained(os.path.join(outdir, "adapter"), safe_serialization=True)
            save_file(
                {f"norms.{k}": v.detach().cpu().contiguous() for k, v in norms.state_dict().items()},
                os.path.join(outdir, "norms.safetensors"),
            )
            tok.save_pretrained(os.path.join(outdir, "tokenizer"))

        summary = {
            "job": job,
            "rows": int(len(df)),
            "prompts": int(df.prompt_id.nunique()),
            "valid_acc": acc,
            "valid_is_dummy": job in ("stage1", "final"),
            "steps": step,
            "skipped": skipped,
            "loss": float(np.mean(losses[-180:])) if losses else None,
            "secs": time.time() - t0,
            "epochs": epochs,
            "init_adapter": init_adapter,
            "model": MODEL_ID,
        }
        with open(done, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        out_vol.commit()
        print(
            f"job {job} done acc {acc:.4f} steps {step} skipped {skipped} ({summary['secs']:.0f}s)",
            flush=True,
        )
        del model, base, opt, norms, plain, centered, tokens
        torch.cuda.empty_cache()
        return summary

    pool = load_frame("/data/pool_v5.parquet", min_models=3)
    host = load_frame("/data/train.csv")
    print(
        f"pool_v5 rows {len(pool)} prompts {pool.prompt_id.nunique()} | host rows {len(host)} gpu {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    results = [
        run_job(pool, "stage1", S1_EPOCHS, S1_LR_ADAPTER, S1_LR_HEAD, True, None),
    ]
    stage1_adapter = os.path.join(out_root, "job_stage1", "adapter")
    if not os.path.isdir(stage1_adapter):
        raise RuntimeError(f"stage 1 adapter missing: {stage1_adapter}")
    for job in S2_JOBS:
        results.append(
            run_job(host, job, S2_EPOCHS, S2_LR_ADAPTER, S2_LR_HEAD, False, stage1_adapter)
        )
    print(results, flush=True)
    return results


@app.local_entrypoint()
def main():
    for path in (POOL_LOCAL, HOST_LOCAL):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
    # spawn returns immediately. modal run --detach then exits, and the H100
    # job keeps running after this laptop process is gone.
    call = train.spawn()
    print(f"spawned {call.object_id}")
