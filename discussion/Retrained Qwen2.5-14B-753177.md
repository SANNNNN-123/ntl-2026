# Retrained the Qwen2.5-14B member of nlm-v27 on the 120k pool: OOF 0.569 → 0.605, public LB 0.651 → 0.647. Why?

Author: Harshita Kumari · 2026-10-27 · 3 votes

https://www.kaggle.com/competitions/name-that-llm/discussion/753177

I tried what looked like the obvious next step after [nlm-v27](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27) (0.651). The result went the wrong way, and I'd like to understand why before spending more compute.

## Background

In nlm-v27, three members (q14v5, q8v5, ll8v5) were retrained on the 120k `nlm_pool_v5`. That step took the notebook from 0.619 to 0.651.

The Qwen2.5-14B member (`q25_14`, Nishant's part-two checkpoint, weight 1.0) was not retrained. According to the author's member table, it was still trained on the old ~35k pool, with OOF 0.569.

## What I did

I retrained only that member, with the same two-stage LoRA24 recipe as the other v5 members:

- **Model:** Qwen2.5-14B base, first 32 of 48 layers, final norm off.
  - LoRA r16 / alpha 32 on q, k, v, o, gate, up and down.
  - Mean pool (first token masked) of layers 3, 5, 8, 11, 16, 21, 27 and 32, with a LayerNorm per layer.
  - A plain head and a prompt-centred head; batches are triplets of answers to the same prompt.
- **Stage 1:** 1 epoch on all of `nlm_pool_v5` (119,899 passages), class-weighted, lr 1e-4 (LoRA) / 1e-3 (heads), cosine.
- **Stage 2:** 2 epochs on the official train, lr 5e-5 / 5e-4.
  - 5 prompt-grouped folds for OOF, plus a full-data fit for inference.
  - max_len 384, `<|endoftext|>` appended.
- **Hardware:** about 2.3 h on one H100.

## Results

| | Old q25_14 (Nishant, ~35k pool) | Retrained (v5 pool) |
| --- | --- | --- |
| Stage 1 only, on all official train | — | 0.570 |
| 5-fold OOF balanced accuracy | 0.569 | **0.605** (folds 0.617 / 0.601 / 0.606 / 0.600 / 0.604) |
| In-sample check inside nlm-v27 | — | 0.95 |

**Leaderboard:** in nlm-v27 I changed one line, `nmeta('lora_qwen25_14b')` → my checkpoint, keeping weight 1.0 and everything else unchanged.

- The log confirms it loaded correctly: `l24 q25_14: adapter ... done (bf16 redo batches 0)`, in-sample acc 0.95.
- **Public LB: 0.647, against 0.651 for the original.**

## My guesses (happy to be corrected)

1. **Noise.** The author reported identical notebooks scoring 0.600 and 0.605, and ATLAS is time-budgeted, so the scored rows can vary between reruns. A −0.4 change may be inside that.
2. **Lost diversity.** The old member saw a different, smaller data mix, so its errors differed from q14v5, q8v5 and ll8v5. Trained on the same v5 pool, the new one may mostly repeat their errors. A stronger but more correlated member could add less than a weaker, independent one.
3. **Blend weights.** The ensemble weights (plus Sinkhorn and ATLAS gating) were chosen with the old member in place.

## Questions

- Has anyone else seen a member improve clearly in OOF but not in the ensemble here? Is there a reliable way to judge "does this help the blend" offline, with only 2,699 rows?
- Is there an estimate of the hidden-rerun noise for nlm-v27 (the same version submitted twice)?
- Would keeping both models (old Nishant q25_14 at 1.0 plus the retrained one at a lower weight) be the better test of the diversity explanation?

Credits: dhruvtibarewal (nlm-v27, v5 pool and adapters), Nishant Dahal (two-stage recipe and part-two checkpoints), nuridudak (122k clean pool), and the LoRA24 recipe authors.

Notebook: https://www.kaggle.com/code/harshitakumari256/nlm-v27?scriptVersionId=362705244

## Comment

**HALX2158** · 2026-10-27 · 0 votes

We measured something close to this with zero-weight ablations on the hidden test. Each run is the same notebook with one member's blend weight set to 0. Rescoring the same predictions gave the same score in our runs (two versions of an nlm-v21 fork both scored 0.617), so these deltas are not rerun noise.

On a part-two-based blend (0.638):

- Removing the old Nishant q25_14, which has OOF 0.569 and only weight 0.2 there, cost -0.014 (plain head alone -0.010). That is about as much as the 0.611-OOF Qwen3-14B member's plain head (-0.017).
- The n-gram SVC (OOF 0.473) cost -0.011.

So a weak but differently trained member carries far more LB weight than its OOF suggests.

On nlm-v27 without Llama (0.642), adding is not the mirror of removing:

- Adding the author's ModernBERT-large (OOF 0.588) at weight 1.5 gave -0.004.
- Our own Mistral-7B two-stage LoRA24 on the same v5 pool (OOF 0.578) at weight 1.0 gave +0.002.

Both fit your diversity and blend-weight guesses. Replacing the one member trained on a different data mix with a v5-pool clone removes exactly what the others don't already have. I'd try keeping the old q25_14 at 1.0 and adding yours at around 0.3-0.5. That is your third question, and I think it is the right test.
