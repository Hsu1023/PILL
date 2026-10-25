from typing import Dict, List, Sequence, Tuple
import torch
from .backends import infer_model_backend

def infer_model_backend(model_name: str | None, model_backend: str | None = None) -> str:
    if model_backend in {"llada", "llada_moe", "illada", "dream"}:
        return model_backend
    normalized_name = (model_name or "").lower()
    if "dream" in normalized_name:
        return "dream"
    if "illada" in normalized_name:
        return "illada"
    if "llada-moe" in normalized_name or "lladamoe" in normalized_name or "a1b" in normalized_name:
        return "llada_moe"
    return "llada"


def build_masked_batch(
    tokenizer,
    problems_batch: Sequence[Dict],
    mask_id: int,
    device: str,
    model_backend: str = "llada",
):
    prompts = [problem["prompt"] for problem in problems_batch]
    suffixes = [problem["suffix"] for problem in problems_batch]

    encoded_prefix = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    encoded_suffix = tokenizer(suffixes, add_special_tokens=False, padding=True, return_tensors="pt")

    prefix_ids = encoded_prefix["input_ids"]
    prefix_mask = encoded_prefix["attention_mask"]
    suffix_ids = encoded_suffix["input_ids"]
    suffix_mask = encoded_suffix["attention_mask"]

    prefix_len = prefix_ids.shape[1]
    suffix_len = suffix_ids.shape[1]
    batch_size = prefix_ids.shape[0]

    input_ids = torch.full((batch_size, prefix_len + 1 + suffix_len), mask_id, dtype=torch.long)
    input_ids[:, :prefix_len] = prefix_ids
    input_ids[:, prefix_len + 1:] = suffix_ids
    attention_mask = torch.cat(
        [
            prefix_mask,
            torch.ones((batch_size, 1), dtype=prefix_mask.dtype),
            suffix_mask,
        ],
        dim=1,
    )

    backend = infer_model_backend(model_name=None, model_backend=model_backend)
    model_attention_mask = attention_mask
    position_ids = None
    probe_positions = torch.full((batch_size,), prefix_len, dtype=torch.long)
    if backend == "dream":
        token_mask = attention_mask.to(torch.long)
        position_ids = token_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(token_mask == 0, 1)
        model_attention_mask = torch.logical_and(
            token_mask.unsqueeze(1).unsqueeze(-2),
            token_mask.unsqueeze(1).unsqueeze(-1),
        )
        if prefix_len > 0:
            prefix_valid = prefix_mask.sum(dim=-1)
            probe_positions = torch.where(
                prefix_valid > 0,
                torch.full_like(probe_positions, prefix_len - 1),
                probe_positions,
            )

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "model_attention_mask": model_attention_mask.to(device),
        "position_ids": position_ids.to(device) if position_ids is not None else None,
        "probe_positions": probe_positions.to(device),
        "prefix_len": prefix_len,
        "suffix_len": suffix_len,
        "prefix_mask": prefix_mask,
        "suffix_mask": suffix_mask,
    }


def resolve_layer_indices(num_hidden_state_tensors: int, layer_specs: Sequence[int]) -> List[Tuple[int, str]]:
    resolved = []
    seen = set()
    for layer_spec in layer_specs:
        layer_idx = layer_spec if layer_spec >= 0 else num_hidden_state_tensors + layer_spec
        if layer_idx < 0 or layer_idx >= num_hidden_state_tensors:
            raise ValueError(
                f"Layer spec {layer_spec} is out of range for {num_hidden_state_tensors} hidden-state tensors."
            )
        if layer_idx in seen:
            continue
        seen.add(layer_idx)
        resolved.append((layer_idx, f"layer_{layer_spec}_idx_{layer_idx}"))
    return resolved


def extract_requested_hidden_states(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    resolved_layers,
    position_ids: torch.Tensor | None = None,
):
    base_model = model.model
    block_group_size = getattr(base_model.config, "block_group_size", 1)
    transformer = getattr(base_model, "transformer", None)
    if block_group_size != 1 or transformer is None or not hasattr(transformer, "blocks"):
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        selected = {layer_idx: hidden_states[layer_idx] for layer_idx, _ in resolved_layers}
        del outputs
        del hidden_states
        return selected

    captured = {}
    handles = []
    blocks = transformer.blocks
    final_hidden_state_index = len(blocks)

    def make_pre_hook(layer_idx):
        def hook(_module, args):
            captured[layer_idx] = args[0]
        return hook

    def make_forward_hook(layer_idx):
        def hook(_module, _args, output):
            captured[layer_idx] = output
        return hook

    for layer_idx, _ in resolved_layers:
        if layer_idx == final_hidden_state_index:
            handles.append(transformer.ln_f.register_forward_hook(make_forward_hook(layer_idx)))
        else:
            handles.append(blocks[layer_idx].register_forward_pre_hook(make_pre_hook(layer_idx)))

    try:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer_idx for layer_idx, _ in resolved_layers if layer_idx not in captured]
    if missing:
        raise RuntimeError(f"Failed to capture hidden states for layers: {missing}")
    return captured


def infer_num_hidden_state_tensors(model) -> int:
    base_model = model.model
    transformer = getattr(base_model, "transformer", None)
    if transformer is not None and hasattr(transformer, "blocks"):
        return len(transformer.blocks) + 1

    layers = getattr(base_model, "layers", None)
    if layers is not None:
        return len(layers) + 1

    config = model.config
    if hasattr(config, "num_hidden_layers"):
        return int(config.num_hidden_layers) + 1
    if hasattr(config, "n_layers"):
        return int(config.n_layers) + 1

    raise AttributeError(
        "Unable to infer number of hidden-state tensors from model. "
        "Expected transformer.blocks, model.layers, config.num_hidden_layers, or config.n_layers."
    )


def pool_context_hidden(
    hidden: torch.Tensor,
    prefix_mask: torch.Tensor,
    suffix_mask: torch.Tensor,
    prefix_len: int,
    suffix_len: int,
    context_pool_size: int,
):
    batch_size, _, hidden_dim = hidden.shape
    pooled_prefix = torch.zeros((batch_size, hidden_dim), dtype=hidden.dtype, device=hidden.device)
    pooled_suffix = torch.zeros((batch_size, hidden_dim), dtype=hidden.dtype, device=hidden.device)
    prefix_missing = torch.zeros(batch_size, dtype=torch.bool, device=hidden.device)
    suffix_missing = torch.zeros(batch_size, dtype=torch.bool, device=hidden.device)

    for batch_idx in range(batch_size):
        prefix_valid = int(prefix_mask[batch_idx].sum().item())
        suffix_valid = int(suffix_mask[batch_idx].sum().item())

        if prefix_valid > 0 and context_pool_size > 0:
            prefix_take = min(prefix_valid, context_pool_size)
            prefix_start = prefix_len - prefix_take
            pooled_prefix[batch_idx] = hidden[batch_idx, prefix_start:prefix_len, :].mean(dim=0)
        else:
            prefix_missing[batch_idx] = True

        if suffix_valid > 0 and context_pool_size > 0:
            suffix_take = min(suffix_valid, context_pool_size)
            suffix_start = prefix_len + 1 + (suffix_len - suffix_valid)
            suffix_end = suffix_start + suffix_take
            pooled_suffix[batch_idx] = hidden[batch_idx, suffix_start:suffix_end, :].mean(dim=0)
        else:
            suffix_missing[batch_idx] = True

    return pooled_prefix, pooled_suffix, prefix_missing, suffix_missing
