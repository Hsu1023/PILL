"""Inference from a prefix and suffix, without reference solutions or datasets."""

from pathlib import Path
from types import SimpleNamespace
import math
import torch
from .probe import MLPRegressor
from .features import (
    build_masked_batch,
    extract_requested_hidden_states,
    infer_num_hidden_state_tensors,
    pool_context_hidden,
)
from .defaults import DEFAULTS


def load_probe(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "pill_probe_v1":
        raise ValueError("Expected a PILL release checkpoint (pill_probe_v1).")
    entry = payload["layer"]
    config = entry["probe_config"]
    probe = MLPRegressor(
        entry["feature_dim"],
        config["hidden_dims"],
        config["dropout"],
        config["missing_context_mode"],
        config["use_context_features"],
    )
    probe.load_state_dict(entry["state_dict"], strict=True)
    return payload, probe.to(device).eval()


@torch.inference_mode()
def predict_lengths(model, tokenizer, probe, payload, problems, mask_id, device):
    if not problems:
        return []
    entry = payload["layer"]
    index = entry["hidden_state_index"]
    if not 0 <= index < infer_num_hidden_state_tensors(model):
        raise ValueError("Probe feature layer is incompatible with the backbone.")
    batch = build_masked_batch(
        tokenizer,
        problems,
        mask_id,
        device,
        payload["model_backend"],
    )
    hidden = extract_requested_hidden_states(
        model,
        batch["input_ids"],
        batch["model_attention_mask"],
        [(index, "probe")],
        batch["position_ids"],
    )[index]
    features = hidden[
        torch.arange(hidden.shape[0], device=hidden.device), batch["probe_positions"]
    ]
    missing_prefix = missing_suffix = torch.zeros(
        len(problems), dtype=torch.bool, device=device
    )
    if entry["probe_config"]["use_context_features"]:
        before, after, missing_prefix, missing_suffix = pool_context_hidden(
            hidden,
            batch["prefix_mask"],
            batch["suffix_mask"],
            batch["prefix_len"],
            batch["suffix_len"],
            entry["probe_config"]["context_pool_size"],
        )
        features = torch.cat([before, features, after], dim=-1)
    z = probe(features.float(), missing_prefix, missing_suffix).cpu()
    values = (
        torch.expm1(z * entry["target_std"] + entry["target_mean"])
        .clamp_min(0)
        .tolist()
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("The probe returned a non-finite length prediction.")
    return values


def predict_length(model, tokenizer, probe, payload, prefix, suffix, mask_id, device):
    return predict_lengths(
        model,
        tokenizer,
        probe,
        payload,
        [{"prompt": prefix, "suffix": suffix}],
        mask_id,
        device,
    )[0]


def candidate_lengths(predicted, radius=2, max_length=None):
    if radius < 0 or (max_length is not None and max_length < 1):
        raise ValueError("radius must be nonnegative and max_length must be positive.")
    center = max(1, round(predicted))
    if max_length is not None:
        center = min(center, max_length)
    return center, [
        n
        for n in range(center - radius, center + radius + 1)
        if n >= 1 and (max_length is None or n <= max_length)
    ]


class PILL:
    def __init__(self, checkpoint, device="cuda", model_path=None):
        from .backends import load_model_and_tokenizer, resolve_runtime_mask_id

        self.device = device
        self.payload, self.probe = load_probe(checkpoint, device)
        dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.model, self.tokenizer, self.backend = load_model_and_tokenizer(
            model_path or self.payload["model_name"],
            device,
            dtype,
            self.payload["model_backend"],
        )
        self.mask_id = resolve_runtime_mask_id(
            None, self.model, self.tokenizer, self.backend
        )
        if self.tokenizer.pad_token_id == self.mask_id:
            raise ValueError("Padding and mask token IDs must differ.")

    @torch.inference_mode()
    def infill(self, prefix, suffix, **options):
        if not isinstance(prefix, str) or not isinstance(suffix, str):
            raise TypeError("prefix and suffix must be strings.")
        predicted = predict_length(
            self.model,
            self.tokenizer,
            self.probe,
            self.payload,
            prefix,
            suffix,
            self.mask_id,
            self.device,
        )
        return self._generate(prefix, suffix, predicted, **options)

    def infill_many(self, records, *, probe_feature_batch_size=1, **options):
        """Yield completions in input order, with configurable probe-feature batching.

        Prefixes and suffixes retain the original independent left padding. Thus
        probe predictions can depend on batch size and input order.
        """
        if probe_feature_batch_size < 1:
            raise ValueError("probe_feature_batch_size must be positive.")
        for start in range(0, len(records), probe_feature_batch_size):
            batch = records[start : start + probe_feature_batch_size]
            if not all(
                isinstance(row.get(key), str)
                for row in batch
                for key in ("prefix", "suffix")
            ):
                raise TypeError("Each record needs string prefix and suffix fields.")
            problems = [
                {"prompt": row["prefix"], "suffix": row["suffix"]} for row in batch
            ]
            predictions = predict_lengths(
                self.model,
                self.tokenizer,
                self.probe,
                self.payload,
                problems,
                self.mask_id,
                self.device,
            )
            for offset, (row, predicted) in enumerate(zip(batch, predictions)):
                yield self._generate(
                    row["prefix"],
                    row["suffix"],
                    predicted,
                    language=row.get("language", "python"),
                    task_id=row.get("task_id", str(start + offset)),
                    **options,
                )

    @torch.inference_mode()
    def _generate(
        self,
        prefix,
        suffix,
        predicted,
        *,
        language="python",
        candidate_radius=2,
        max_length=None,
        steps=None,
        temperature=0.0,
        syntax_filter=False,
        task_id="example",
        alpha=None,
        suffix_head_length=None,
    ):
        from .selection import run_generation_batch
        from .dream import run_generation_batch_dream

        if not isinstance(prefix, str) or not isinstance(suffix, str):
            raise TypeError("prefix and suffix must be strings.")
        if steps is not None and steps < 1:
            raise ValueError("steps must be positive.")
        if temperature < 0 or (alpha is not None and not 0 <= alpha <= 1):
            raise ValueError(
                "temperature must be nonnegative and alpha must be in [0, 1]."
            )
        if suffix_head_length is not None and suffix_head_length < 0:
            raise ValueError("suffix_head_length must be nonnegative.")
        center, lengths = candidate_lengths(predicted, candidate_radius, max_length)
        args = SimpleNamespace(**DEFAULTS)
        args.mask_id, args.middle_mask_len, args.front_mask_lengths = (
            self.mask_id,
            center,
            lengths,
        )
        args.steps, args.temperature = steps, temperature
        args.posthoc_filter_invalid_ast = syntax_filter
        args.posthoc_full_suffix_context = self.payload["task"] == "text" or (
            self.backend == "dream" and self.payload["task"] == "multilingual-code"
        )
        args.posthoc_select_alpha = (
            alpha if alpha is not None else (0.9 if self.backend == "dream" else 0.5)
        )
        args.posthoc_suffix_head_length = (
            suffix_head_length
            if suffix_head_length is not None
            else (8 if "Dream-Coder" in self.payload["model_name"] else 4)
        )
        problem = {
            "task_id": task_id,
            "prompt": prefix,
            "suffix": suffix,
            "language": language,
        }
        if self.backend == "dream":
            samples = run_generation_batch_dream(
                self.model,
                self.tokenizer,
                self.device,
                [problem],
                center,
                lengths,
                args,
            )
        else:
            samples = run_generation_batch(
                self.model, self.tokenizer, self.device, [problem], center, args
            )
        result = samples[0]
        return {
            "task_id": task_id,
            "completion": result["completion"],
            "prediction": predicted,
            "candidate_lengths": lengths,
            "selected_length": result["mask_token_length"],
            "candidates": result["all_front_completions"],
        }
