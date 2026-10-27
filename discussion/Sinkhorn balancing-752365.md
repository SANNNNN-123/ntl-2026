# Sinkhorn balancing: what it costs when the test mix is uneven

Author: s0chn1y · 2026-10-24 · 0 votes

https://www.kaggle.com/competitions/name-that-llm/discussion/752365

Many public notebooks end with Sinkhorn balancing: the test probabilities are rescaled until every class gets the same total mass. For balanced accuracy this is a sensible correction if the test is close to even. We do not know the hidden test's mix, so we measured what balancing costs when the mix is uneven.

**Setup.**

- Out-of-fold probabilities of a strong blend: 2,699 passages, prompt-grouped 5-fold CV, about 0.64 balanced accuracy.

- Test pools of 2,000 passages, the model mix drawn from a symmetric Dirichlet(alpha), 40 pools per setting. Alpha 1 is a very uneven mix, with some models nearly absent. Alpha 100 is nearly even.

- On each pool: the plain argmax, and Sinkhorn at 1, 3 and 50 iterations.

| test mix | plain | 1 iteration | 3 iterations | 50 iterations |
| --- | --- | --- | --- | --- |
| exactly even (all OOF rows) | 0.6387 | 0.6398 | 0.6409 | 0.6379 |
| alpha 100, nearly even | 0.6384 | 0.6391 | 0.6395 | 0.6363 |
| alpha 20 | 0.6357 | 0.6376 | 0.6379 | 0.6298 |
| alpha 5 | 0.6393 | 0.6391 | 0.6384 | 0.6167 |
| alpha 1, very uneven | 0.6361 | 0.6345 | 0.6284 | 0.5704 |

Small pools are worse still. On a sparse CV scheme (about 75 passages per pool, 1 to 4 per prompt, uneven mix), full balancing took the same blend from 0.645 to 0.475.

**What we take from it.**

- **Full balancing is a bet that the test is close to even.** If it is, you gain a few thousandths. If it is not, you can lose several points.

- **A partial balance is a cheap hedge.** One to three iterations keep most of the gain on an even test and lose under 0.01 on the most uneven mix we tried.

- **OOF predictions cannot show the gain some people see on the LB.** OOF rows carry no shift between train and test, so this measures the risk only. If balancing helps you on the public LB, that says your test predictions lean towards some models. It says nothing about the test's mix.

To check your own blend, take its OOF log-probabilities and labels:

```
import numpy as np
from sklearn.metrics import balanced_accuracy_score

def sinkhorn(logp, iters):
    """Rescale log-probabilities until every class gets the same total mass; rows stay normalised."""
    z, target = logp.copy(), np.log(len(logp) / logp.shape[1])
    for _ in range(iters):
        m = z.max(0, keepdims=True)
        z = z - (m + np.log(np.exp(z - m).sum(0, keepdims=True))) + target
        z = z - z.max(1, keepdims=True)
        z -= np.log(np.exp(z).sum(1, keepdims=True))
    return z

def uneven_pools(y, alpha, size=2000, draws=40, seed=0):
    """Row indices of test pools whose class mix is drawn from a symmetric Dirichlet(alpha)."""
    rng, classes = np.random.default_rng(seed), np.unique(y)
    by_class = [np.flatnonzero(y == c) for c in classes]
    for _ in range(draws):
        counts = rng.multinomial(size, rng.dirichlet([alpha] * len(classes)))
        yield np.concatenate([rng.choice(ix, k) for ix, k in zip(by_class, counts) if k])

# logp: (n, 18) OOF log-probabilities, y: (n,) labels, labels: class names in column order
for alpha in [1, 5, 20, 100]:
    scores = np.mean([[balanced_accuracy_score(y[r], labels[sinkhorn(logp[r], k).argmax(1)]) for k in (0, 1, 3, 50)]
                      for r in uneven_pools(y, alpha)], axis=0)
    print(alpha, scores.round(4))

```


## Comments

### GoPain · 2026-10-24 · 0 votes

Thanks, this matches what I see. One addition: what Sinkhorn removes is a per-class offset in the scores, so the same gain can be learned on OOF and applied to test without any assumption about the test mix.

Cross-fitted offsets: learn one log-offset per class on four folds' OOF (Sinkhorn on those rows, keep the column scalings), add them to the fifth fold's scores, then argmax. My grouped 5-fold OOF (2,699 rows):

|  | argmax | Sinkhorn, all rows | cross-fitted offsets |
| --- | --- | --- | --- |
| DeBERTa-v3-large, two-stage | 0.521 | 0.564 | 0.560 |
| Qwen3-8B LoRA | 0.595 | 0.600 | 0.599 |
| LoRA + 0.5 DeBERTa + 1.0 n-gram SVC | 0.626 | 0.626 | 0.621 |

- A single member with strong class preferences (the DeBERTa here) gets almost all of its Sinkhorn gain from offsets learned on OOF. Offsets learned that way do not care what the test mix is.

- On a blend whose members lean in different directions, little is left to remove: plain argmax already equals Sinkhorn on OOF. Cross-fitted offsets only add noise (−0.005). Full balancing then buys nothing on CV and keeps the risk you measured.

- On the public LB, Sinkhorn on an earlier LoRA alone gave +0.011 (0.538 to 0.549) while its OOF did not move (0.5760 to 0.5761). That fits your last point: the LB gain reflects how test predictions lean, not the test's mix.

I checked the same comparison on a known-truth simulation (18 Gaussian classes with per-class offsets). On a balanced test, offsets from validation are within 0.003 of Sinkhorn. With a Dirichlet(1) test mix, Sinkhorn's regret against the Bayes rule rises to 0.060 while the offsets stay at 0.018. Estimating the test prior by EM and balancing towards it was worse than both. That is expected, because for balanced accuracy the optimal decision does not depend on the test prior. Notebook with the code and tables: [https://www.kaggle.com/code/locdunhat/sinkhorn-rebalancing-under-balanced-accuracy](https://www.kaggle.com/code/locdunhat/sinkhorn-rebalancing-under-balanced-accuracy)
