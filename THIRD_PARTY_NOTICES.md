# Third-party notices

This release retains model and inference components from the research implementation on which PILL was developed.

- LLaDA components retain the original MIT notice, copyright 2025 NieShenRuc, in `licenses/LLaDA-MIT.txt`.
- Dream model, configuration, and generation files retain their Apache-2.0 headers crediting the Dream team, HKUNLP Group, Hugging Face, and other contributors named in those files. The Apache-2.0 license is included in `licenses/Apache-2.0.txt`.
- The small language/syntax helpers extracted from HumanEval infilling retain the OpenAI MIT notice in `licenses/HumanEval-MIT.txt`. Benchmark execution and datasets are not included.
- The LLaDA-MoE model/configuration implementation is retained from the research project. The [upstream model card](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Base) identifies Apache-2.0; that license is included in `licenses/Apache-2.0.txt`.

PILL-specific extraction and changes include separating inference from experiment scripts, packaging model imports locally, exporting inference-only probe checkpoints, and adding the prefix/suffix API. The Dream generation helper is packaged locally with the unused EOS-penalty option removed. Original model file headers are retained.
