<h1>Predict, Don't Iterate: Efficient Adaptive-Length Infilling for Diffusion Language Models</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.02108"><img src="https://img.shields.io/badge/arXiv-2609.02108-b31b1b?logo=arxiv" alt="arXiv paper"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/Code%20License-MIT-eab308" alt="MIT code license"></a>
</p>


**Accepted at EMNLP 2026 (Main Conference).**

PILL predicts an infill length from the base model's hidden states, generates candidates around that length, and selects a completion using completion and suffix scores. Inputs are a **prefix and suffix**; no reference completion is required.

This repository provides the probe architecture, pretrained weights, feature extraction, candidate generation, and selection. Probe training, training data preparation, baseline implementations, and ablation experiments are outside this release.

## 🔧 Installation

Use Python 3.10 or newer. The dependency versions below match the environment used for local validation.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from the repository root. A GPU with sufficient memory for the selected backbone and candidate sequence is recommended. Memory requirements depend on context length and candidate lengths; this release does not specify a measured minimum. CPU execution is available with `--device cpu`.

## 📦 Pretrained probes

Each probe belongs to one **base model and task family**. Base-model weights are downloaded separately through Transformers; the files below contain only the length probe.

| Model directory | Base model | Available tasks |
| --- | --- | --- |
| `llada-base` | `GSAI-ML/LLaDA-8B-Base` | `python-single`, `python-multi`, `multilingual-code`, `text` |
| `llada-instruct` | `GSAI-ML/LLaDA-8B-Instruct` | `python-single`, `python-multi`, `multilingual-code`, `text` |
| `dream` | `Dream-org/Dream-v0-Base-7B` | `python-single`, `python-multi`, `multilingual-code`, `text` |
| `dreamcoder` | `Dream-org/Dream-Coder-v0-Base-7B` | `python-single`, `python-multi`, `multilingual-code` |
| `llada-moe` | `inclusionAI/LLaDA-MoE-7B-A1B-Base` | `python-single`, `python-multi`, `multilingual-code`, `text` |

`multilingual-code` corresponds to the Java/C++ task family. `text` corresponds to natural-language infilling. The same probe can serve multiple benchmarks within its task family. There are 19 probes, approximately 393 MiB total.

Weights are distributed as GitHub Release assets rather than committed to Git. The [manifest](checkpoints/manifest.json) records model IDs, asset names, sizes, and SHA-256 checksums. Select a published tag from [Releases](https://github.com/Hsu1023/PILL/releases), then run:

```bash
python scripts/download_weights.py --list
python scripts/download_weights.py llada-base/python-single --tag <release-tag>
```

The downloader verifies the checksum and writes `checkpoints/llada-base/python-single.pt`. Local release preparations may already contain these files. The release checkpoint format is `pill_probe_v1`; research checkpoints are not accepted by this entry point.

## 🚀 Quick start

```bash
python inference.py \
  --checkpoint checkpoints/llada-base/python-single.pt \
  --input examples/python.jsonl \
  --output outputs/python.jsonl \
  --syntax-filter
```

Each input line is a JSON object:

```json
{"task_id":"add","prefix":"def add(a, b):\n    ","suffix":"\n","language":"python"}
```

Only `prefix` and `suffix` are required. `task_id` and `language` are optional. Output records contain the selected `completion`, predicted length (`prediction`), `candidate_lengths`, `selected_length`, and all `candidates` in matching order.

For text infilling:

```bash
python inference.py \
  --checkpoint checkpoints/llada-base/text.pt \
  --input examples/text.jsonl \
  --output outputs/text.jsonl
```

The backbone is selected from checkpoint metadata. Use `--model-path /path/to/model` only for a local copy of that exact base model. The Python API is also available:

```python
from pill.inference import PILL

engine = PILL("checkpoints/llada-base/python-single.pt", device="cuda")
result = engine.infill("def add(a, b):\n    ", "\n", syntax_filter=True)
print(result["completion"])
```

## ⚙️ Inference settings

The default candidate radius is 2, producing up to five candidate lengths around the rounded prediction. Lengths are at least one token. `--max-length` optionally caps the predicted center and candidate lengths. Without `--steps`, decoding uses the maximum candidate/center length.

The release uses interpolated positions, completion/suffix mean-log-probability selection, and four sparse suffix positions selected from a 32-token window at attention layer -2, excluding low-information tokens. Selection uses `alpha * completion_score + (1-alpha) * suffix_score` when both are available.

Defaults are alpha 0.5 for LLaDA models and 0.9 for Dream models, with four suffix-head tokens (eight for DreamCoder). `--alpha` and `--suffix-head-length` override those defaults. Full-suffix visibility is enabled for text probes and Dream-family multilingual-code probes. These are convenient inference presets based on the original run scripts, **not a claim that one preset reproduces every reported benchmark configuration**.

`--syntax-filter` enables the original syntax-based candidate filter. Python uses `ast.parse`; C++ requires `g++`, and Java requires `javac`. Inputs should include the surrounding source needed for a complete program. If every candidate fails syntax validation, the original fallback selection is retained. Syntax validation does not establish functional correctness. Omit this option for natural-language inputs.

Other controls: `--temperature` (default 0), `--seed` (default 42), `--steps`, and `--candidate-radius`. The CLI processes generation one record at a time while generating multiple candidate lengths within each record.

Probe feature extraction uses batch size 1 by default. To process multiple inputs through the Python API, use `engine.infill_many(records, probe_feature_batch_size=1, syntax_filter=True)`.

## 📂 Layout

```text
pill/
  inference.py       # public API and length prediction
  probe.py           # MLP architecture
  features.py        # hidden states and context pooling
  backends.py        # backbone loading
  decoding.py        # parallel masked decoding and confidence scoring
  selection.py       # suffix selection and candidate ranking
  dream.py           # Dream-family inference
  models/            # backbone implementations used by inference
checkpoints/
  manifest.json      # released weight catalog and checksums
examples/            # small input examples
scripts/             # weight downloader
```

## 🧪 Validation

```bash
python -m unittest discover -s tests -v
```

Tests use tiny randomly initialized LLaDA, Dream, and LLaDA-MoE models on CPU to exercise feature extraction and candidate generation/selection. They do not measure pretrained-model quality or GPU memory use. All 19 pretrained probes across the five backbones were also smoke-tested on an NVIDIA A100-SXM4-40GB. For one short input per probe, predicted lengths, all generated candidates, and selected completions matched the original implementation. These checks used a 32-token length cap and syntax filtering for code inputs; they are not full benchmark reproduction.

## 📜 License and attribution

See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Retained third-party notices apply to their respective files. Base models are obtained separately and remain subject to their own terms.

## 📝 Citation

```bibtex
@inproceedings{xu2026predict,
  title={Predict, don’t iterate: Efficient adaptive-length infilling for diffusion language models},
  author={Xu, Haobo and Chen, Sirui and Bei, Yuanchen and Chen, Lingjie and Yan, Yuchen and Fu, Dongqi and He, Jingrui and Tong, Hanghang},
  booktitle={Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  pages={24171--24190},
  year={2026}
}
```
