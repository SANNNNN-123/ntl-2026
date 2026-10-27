# Name That LLM

Ask 18 frontier language models the same question, and no two answers read quite the same. Even in a single short paragraph, every model leaves subtle clues in its word choice, tone, and sentence structure.

In this competition, you'll identify which among 18 leading models wrote each unlabeled passage. Your work will sharpen methods for AI provenance and benchmark auditing.

### Evaluation

Submissions are evaluated on **balanced accuracy**: the average, over all candidate models, of the fraction of that model's test passages you labelled correctly (per-class recall).

$$
\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \frac{|\{i : y_i = k \land \hat{y}_i = k\}|}{|\{i : y_i = k\}|}
$$

where $K = 18$ is the number of candidate models, $y_i$ is the true authoring model and $\hat{y}_i$ is your prediction for passage $i$.

The number of test passages per model is not equal, and the mix of models differs between the public and private leaderboard. Balanced accuracy gives every model the same weight.

### Valid Labels

Every prediction must be exactly one of the following 18 model names:

| Provider | Model Name |
| --- | --- |
| OpenAI | `gpt-5.5` |
| OpenAI | `gpt-5.6-luna` |
| OpenAI | `gpt-5.6-sol` |
| OpenAI | `gpt-5.6-terra` |
| OpenAI | `gpt-6-astra` |
| Google | `gemini-3.1-pro-preview` |
| Google | `gemini-3.6-flash` |
| Google | `gemini-3.7-flash` |
| Google | `gemini-3.8-flash` |
| Anthropic | `claude-fable-5` |
| Anthropic | `claude-opus-4-6` |
| Anthropic | `claude-opus-4-8` |
| Anthropic | `claude-opus-5` |
| Anthropic | `claude-sonnet-5` |
| Qwen | `qwen3.8-max` |
| ZAI | `glm-5.3` |
| ZAI | `glm-5.3-flash` |
| Moonshot AI | `kimi-k3` |

### Submission File

This is a **Code Competition**. Your notebook reads the test set from `/kaggle/input/competitions/name-that-llm/test.csv` and must write a file named `submission.parquet` to the notebook's working directory.

During an interactive session `test.csv` contains a few example rows so your notebook can run end to end. When your notebook is submitted for scoring, it is re-run against the full hidden test set (~4000 passages).

For each `id` in the test set, predict the authoring `model`. Every `id` must appear exactly once. The file should contain a header and have the following format:

```text
id,model
a1b2c3d4e5f6,gpt-6-astra
b2c3d4e5f6a7,claude-opus-5
c3d4e5f6a7b8,gemini-3.8-flash
...

```
