# Qwen3-8B-base-Lora — ELI5 notes

Guide to [`Qwen3-8B-base-Lora.ipynb`](Qwen3-8B-base-Lora.ipynb): what it is, why it’s built this way, and what each part does.

Related public recipe: [`ntl-lora24-fivefold-training.ipynb`](ntl-lora24-fivefold-training.ipynb) (same LoRA24 *idea*, not a line-for-line copy).

**LB results:** see [`experiments.md`](experiments.md) — host-only **0.489**; two-stage extended→host **0.487**.

---

## Big picture

**Competition job:** guess which of 18 LLMs wrote a short passage.

**This notebook:** train a “fingerprint” model (LoRA on Qwen3-8B-Base). It is a **training kernel**, not a real leaderboard submission (it writes an empty `submission.parquet` placeholder).

**Same family as the public LoRA24 notebook:**

| Piece | Value |
| --- | --- |
| Base | Qwen3-8B-Base truncated to **24** layers |
| LoRA | r=16, α=32, dropout=0.05 on q,k,v,o,gate,up,down |
| Pool layers | 2 / 4 / 6 / 8 / 12 / 16 / 20 / 24 |
| Heads | plain + centered Linear classifiers |
| Microbatches | 3 answers to the **same** prompt |
| Loss | CE(plain) + CE(centered) |
| Optim | AdamW 1e-4 (adapters) / 1e-3 (heads), wd 0.01 |
| Schedule | 10% warmup + cosine, 3 epochs, accum 6 |
| Seq length | max_len 447 + EOS |

**How ours differs from the public notebook:**

- Training seed **7** (public uses **42**)
- Simpler fold rule: shuffle prompts with `FOLD_SEED=0`, then `i % 5` (public uses a precomputed nested plan)
- Leaner code (no smoke tests / heavy evaluate script)
- Outputs under `lora_out/job_*`

---

## 5-fold prompt split

We do **not** randomly split individual rows into train/valid.

Each row belongs to a **prompt** (question) via `prompt_id`. We:

1. List unique prompts  
2. Shuffle them **once** with fixed seed `FOLD_SEED = 0` (repeatable)  
3. Put them into **5 buckets** (folds 0–4)

Then for each fold job:

- **Train** on the other 4 buckets  
- **Validate** on the held-out bucket  

So train and valid **never share the same question**. That stops the model from cheating on topic/wording of a question it already saw.

Plus a **`final`** job: train on **all** prompts (adapter for real inference).

---

## Why `worker.py`?

The notebook is **one** Python process. We have **two T4 GPUs** and want them training **different folds at the same time**.

So we:

1. Write training code into `worker.py` (`%%writefile` / `%%writefile -a` cells)  
2. Launch **two processes**, each with its own `CUDA_VISIBLE_DEVICES` and job list  

Each process = its own model in memory = its own GPU.

`cfg` and `jobs` are passed on the command line from the launcher so:

- both GPUs share the same hyperparameters/paths (`cfg`)  
- each GPU gets different work (`jobs`) without two nearly identical scripts  

---

## Hyperparameters cell

| Name | Role |
| --- | --- |
| `SEED` | Training randomness (init, batch order, …). Ours = **7** |
| `FOLD_SEED` | How prompts are shuffled into 5 folds. Ours = **0** |
| `JOBS` | Which GPU runs which jobs (see below) |
| `LAYERS` | Which hidden layers to mean-pool |
| `KEEP` | Keep first N layers of Qwen (24) |
| `MAX_LEN` | Token truncate length (447) before EOS |
| `EPOCHS` | Fixed 3 — no early stopping on valid |
| `ACCUM` | Gradient accumulation steps (6) → effective batch ≈ 3×6 = 18 |
| `LR_ADAPTER` / `LR_HEAD` | Learning rates for LoRA vs heads/norms |
| `WD` / `WARMUP` | Weight decay; warmup fraction of steps |
| `R` / `ALPHA` / `LDROP` | LoRA rank, alpha, dropout |

### `JOBS` dict

```python
JOBS = {
    0: ['0', '2', '4'],       # GPU 0
    1: ['1', '3', 'final'],  # GPU 1
}
```

| Key | Meaning |
| --- | --- |
| `0` | GPU 0 runs fold jobs 0 → 2 → 4 (one after another) |
| `1` | GPU 1 runs fold jobs 1 → 3 → then `final` |

---

## Worker sections (what gets written into `worker.py`)

### 1. Imports, data, folds, tokenize

Creates/overwrites `worker.py` with startup code (does **not** train yet).

When a GPU process later runs the file:

1. Hide Kaggle `torchao` so `peft` imports cleanly  
2. Read `cfg` + `jobs` from argv  
3. Load `train.csv`  
4. Map model names → class ids (`y`)  
5. Build the 5-fold prompt split → `fold`  
6. Tokenize all passages once → `tokens`  
7. Remember pool layers `L` and `cuda:0` (this process’s only GPU)

### 2. `triplets()` — microbatch helper

Builds packs of **3 answers to the same question**.

Why same prompt? The **centered** head subtracts the mean of the 3 fingerprints. That only makes sense if they share a question (same topic, different model styles).

**Example:** prompt Q1 has answers at rows `[10,11,12,13,14]`.

1. Shuffle → e.g. `[10,14,11,13,12]`  
2. Chunk by 3 → `[10,14,11]` and leftover `[13,12]`  
3. Pad short chunk to 3 using another Q1 row → `[13,12,10]`  
4. Do for every prompt; shuffle the list of triplets  
5. Pad list length to a multiple of `ACCUM` (6) by duplicating triplets  

Each inner list of 3 = one microbatch in the train loop.

### 3. Job setup: model, LoRA, heads, optimizer

For each job this GPU was assigned:

1. Choose `fit` / `valid` rows (fold hold-out, or all rows if `final`)  
2. Load Qwen3-8B-Base, keep 24 layers, fp16 + grad checkpointing  
3. Attach LoRA adapters  
4. Build `norms` + `plain` + `centered` heads (+ dropout)  
5. AdamW (two LRs), pre-build epoch triplet `plan`, warmup+cosine schedule, GradScaler  

No weight updates yet — workshop setup only.

### 4. `rep()` + CE train steps

**CE = Cross-Entropy** — standard “how wrong was my class guess?” loss for 18-way classification.

**`rep(idx)`:** run texts through the model → mean-pool selected layers (mask pads + first token) → LayerNorm each → concatenate → fingerprint vector `z`.

**Train loop** for each triplet:

```text
loss = CE(plain(z), true_labels)
     + CE(centered(z - mean(z)), true_labels)
```

| Head | Idea |
| --- | --- |
| plain | Who wrote this, from absolute style? |
| centered | Who wrote this, after removing what’s common to this question? |

Accumulate grads over 6 microbatches → clip → optimizer step. Print loss every 30 steps.

### 5. Eval + save artifacts

After the fixed 3 epochs (no mid-training checkpoints):

1. Fingerprint valid rows → matrix `Z`  
2. Save under `lora_out/job_<id>/`  
3. Log `valid_acc` (plain head)  
4. Free GPU memory for the next job  

#### OOF (out-of-fold)

**OOF** = features/scores for a row from a model that **never trained on that row’s prompt**.

| Job | Saves | For |
| --- | --- | --- |
| `job_0` … `job_4` | `valid_features.npz` + `readouts.npz` + `COMPLETE.json` | Fair train-side features for stacking / blend weights — **not** main inference weights |
| `job_final` | same + **`adapter/`**, **`norms.safetensors`**, **`tokenizer/`** | Model you load for real inference |

**ELI5:** fold jobs = practice tests graded fairly on every homework question; `final` = the exam model you ship.

If you only scored train with `final`, every prompt was seen in training → leaky / too optimistic for blending.

---

## Launch cell: locate data & start workers

1. Find `train.csv` (Kaggle paths or local `data/`)  
2. Find Qwen3-8B-Base under `/kaggle/input`  
3. Pack hyperparameters into `cfg`  
4. Start two `python worker.py <cfg> <jobs>` processes  
5. Wait; print logs + summary  
6. Write empty `submission.parquet` (placeholder)

---

## Reading `completed jobs` output

Example fields from each `COMPLETE.json`:

| Field | Meaning |
| --- | --- |
| `job` | `'0'`…`'4'` or `'final'` |
| `valid_acc` | Validation accuracy of the **plain** head on that job’s valid rows |
| `steps` | Optimizer updates (folds ≈ 360; `final` ≈ 450 — more data) |
| `skipped` | GradScaler rejected steps (fp16 overflow). `0` = clean |
| `secs` | Time for **that** job on its GPU |
| `total wall time` | Until **both** workers finished (parallel, so less than sum of `secs`) |

**`valid_acc` detail:** after training, predict model id on held-out fold rows; fraction correct.

- Folds `0`–`4`: meaningful OOF-ish accuracy  
- `final`: uses 3 dummy valid rows → **ignore** that accuracy; keep the adapter  

---

## Mental model (one paragraph)

Shuffle questions into 5 buckets. Train five fold models so every train passage gets an honest fingerprint (OOF). Train one `final` model on everything for inference. Run fold/final jobs as separate GPU processes driven by `worker.py`. Save features/heads per fold and the full adapter for `final`. Blend later using OOF; predict test with `final` (or an ensemble whose weights were fit on OOF).
