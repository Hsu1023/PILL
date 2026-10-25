import inspect
import torch
from torch.nn import functional as F
import numpy as np

def _model_accepts_attention_bias(model):
    cached = getattr(model, "_mp_test_accepts_attention_bias", None)
    if cached is not None:
        return bool(cached)
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        accepts = False
    else:
        accepts = "attention_bias" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    try:
        setattr(model, "_mp_test_accepts_attention_bias", bool(accepts))
    except Exception:
        pass
    return bool(accepts)


def _call_model_logits(model, input_ids, attention_mask=None, position_ids=None, attention_bias=None):
    kwargs = {
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }
    if attention_bias is not None and _model_accepts_attention_bias(model):
        kwargs["attention_bias"] = attention_bias
    return model(input_ids, **kwargs).logits


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _build_parallel_candidate_causal_attention_bias(prefix_len, slot_lengths, device):
    seq_len = prefix_len + sum(slot_length * 2 for slot_length in slot_lengths)
    allow = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    if prefix_len > 0:
        allow[:prefix_len, :prefix_len] = True

    cursor = prefix_len
    for slot_length in slot_lengths:
        text_start = cursor
        text_end = text_start + slot_length
        mask_start = text_end
        mask_end = mask_start + slot_length

        if prefix_len > 0:
            allow[text_start:text_end, :prefix_len] = True
            allow[mask_start:mask_end, :prefix_len] = True

        allow[text_start:text_end, text_start:text_end] = torch.tril(
            torch.ones((slot_length, slot_length), dtype=torch.bool, device=device)
        )
        allow[mask_start:mask_end, text_start:text_end] = torch.tril(
            torch.ones((slot_length, slot_length), dtype=torch.bool, device=device),
            diagonal=-1,
        )

        cursor = mask_end

    attention_bias = torch.full(
        (seq_len, seq_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=device,
    )
    attention_bias.masked_fill_(allow, 0.0)
    return attention_bias


def _build_left_padded_segment_position_ids(token_mask, start_positions, dtype):
    batch_size, segment_len = token_mask.shape
    if segment_len == 0:
        return torch.empty((batch_size, 0), device=token_mask.device, dtype=dtype)

    start_positions = start_positions.to(device=token_mask.device, dtype=dtype)
    token_mask_long = token_mask.to(torch.long)
    ordinal = token_mask_long.cumsum(dim=-1) - 1
    ordinal = ordinal.clamp_min(0).to(dtype)
    position_ids = ordinal + start_positions.unsqueeze(-1)
    return torch.where(token_mask.bool(), position_ids, torch.zeros_like(position_ids))


def _extract_left_padded_suffix_head_ids(suffix_ids, suffix_attention_mask, head_length):
    if suffix_ids is None or head_length <= 0:
        return None, None

    batch_size, suffix_padded_len = suffix_ids.shape
    if suffix_padded_len == 0:
        empty_ids = torch.empty((batch_size, 0), dtype=suffix_ids.dtype, device=suffix_ids.device)
        empty_mask = torch.empty((batch_size, 0), dtype=torch.long, device=suffix_ids.device)
        return empty_ids, empty_mask
    device = suffix_ids.device
    if suffix_attention_mask is None:
        valid_length = torch.full((batch_size,), suffix_padded_len, dtype=torch.long, device=device)
        suffix_attention_mask = torch.ones_like(suffix_ids, dtype=torch.long, device=device)
    else:
        valid_length = suffix_attention_mask.to(torch.long).sum(dim=-1)

    offsets = torch.arange(head_length, device=device, dtype=torch.long).unsqueeze(0)
    start_idx = (suffix_padded_len - valid_length).unsqueeze(-1)
    gather_idx = (start_idx + offsets).clamp(min=0, max=max(suffix_padded_len - 1, 0))
    head_ids = torch.gather(suffix_ids, dim=1, index=gather_idx)
    head_mask = (offsets < valid_length.unsqueeze(-1)).to(dtype=suffix_attention_mask.dtype)
    return head_ids, head_mask


def _extract_left_padded_full_suffix_ids(suffix_ids, suffix_attention_mask, min_length=0):
    if suffix_ids is None:
        return None, None

    suffix_padded_len = suffix_ids.shape[1]
    if suffix_padded_len == 0:
        empty_ids = torch.empty((suffix_ids.shape[0], 0), dtype=suffix_ids.dtype, device=suffix_ids.device)
        empty_mask = torch.empty((suffix_ids.shape[0], 0), dtype=torch.long, device=suffix_ids.device)
        return empty_ids, empty_mask

    if suffix_attention_mask is None:
        valid_length = suffix_padded_len
    else:
        valid_length = int(suffix_attention_mask.to(torch.long).sum(dim=-1).max().item())
    full_length = max(int(min_length), valid_length)
    return _extract_left_padded_suffix_head_ids(
        suffix_ids=suffix_ids,
        suffix_attention_mask=suffix_attention_mask,
        head_length=full_length,
    )


def _build_visible_suffix_window(
    suffix_ids,
    suffix_attention_mask,
    selected_positions,
    mask_id,
    batch_size,
    token_dtype,
    mask_dtype,
    device,
    full_suffix_context=False,
):
    selected_positions = [int(pos) for pos in selected_positions if int(pos) >= 0]
    selected_suffix_count = len(selected_positions)
    selected_positions_tensor = torch.tensor(selected_positions, dtype=torch.long, device=device)

    if full_suffix_context:
        full_suffix_ids, full_suffix_token_mask = _extract_left_padded_full_suffix_ids(
            suffix_ids=suffix_ids,
            suffix_attention_mask=suffix_attention_mask,
            min_length=max(selected_positions) + 1 if selected_positions else 0,
        )
        if full_suffix_ids is None:
            dense_suffix_ids = torch.empty((batch_size, 0), dtype=token_dtype, device=device)
            dense_suffix_token_mask = torch.empty((batch_size, 0), dtype=mask_dtype, device=device)
        else:
            dense_suffix_ids = full_suffix_ids.to(device=device, dtype=token_dtype)
            dense_suffix_token_mask = full_suffix_token_mask.to(device=device, dtype=mask_dtype)
        if selected_suffix_count > 0 and dense_suffix_ids.shape[1] > 0:
            selected_suffix_ids = torch.index_select(dense_suffix_ids, dim=1, index=selected_positions_tensor)
            selected_suffix_valid_mask = torch.index_select(
                dense_suffix_token_mask.bool(),
                dim=1,
                index=selected_positions_tensor,
            )
        else:
            selected_suffix_ids = torch.empty((batch_size, 0), dtype=token_dtype, device=device)
            selected_suffix_valid_mask = torch.empty((batch_size, 0), dtype=torch.bool, device=device)
        return (
            dense_suffix_ids,
            dense_suffix_token_mask,
            selected_suffix_ids,
            selected_suffix_valid_mask,
            selected_positions_tensor,
        )

    dense_suffix_len = (max(selected_positions) + 1) if selected_positions else 0
    dense_suffix_ids = torch.full(
        (batch_size, dense_suffix_len),
        int(mask_id),
        dtype=token_dtype,
        device=device,
    )
    dense_suffix_token_mask = torch.zeros(
        (batch_size, dense_suffix_len),
        dtype=mask_dtype,
        device=device,
    )
    selected_suffix_ids = torch.full(
        (batch_size, selected_suffix_count),
        int(mask_id),
        dtype=token_dtype,
        device=device,
    )
    selected_suffix_valid_mask = torch.zeros(
        (batch_size, selected_suffix_count),
        dtype=torch.bool,
        device=device,
    )

    suffix_padded_len = suffix_ids.shape[1] if suffix_ids is not None else 0
    if suffix_ids is not None and suffix_padded_len > 0:
        if suffix_attention_mask is None:
            valid_length = torch.full((batch_size,), suffix_padded_len, dtype=torch.long, device=device)
        else:
            valid_length = suffix_attention_mask.to(device=device, dtype=torch.long).sum(dim=-1)
        start_idx = (suffix_padded_len - valid_length).unsqueeze(-1)
        for rel_pos in range(dense_suffix_len):
            within_actual = rel_pos < valid_length
            dense_suffix_token_mask[:, rel_pos] = within_actual.to(dtype=mask_dtype)
        for selected_idx, rel_pos in enumerate(selected_positions):
            rel_pos_tensor = torch.full((batch_size, 1), int(rel_pos), dtype=torch.long, device=device)
            gather_idx = (start_idx + rel_pos_tensor).clamp(min=0, max=max(suffix_padded_len - 1, 0))
            gathered_ids = torch.gather(suffix_ids.to(device=device), dim=1, index=gather_idx).squeeze(-1)
            valid_mask = rel_pos < valid_length
            gathered_ids = gathered_ids.to(dtype=token_dtype)
            selected_suffix_ids[:, selected_idx] = torch.where(
                valid_mask,
                gathered_ids,
                selected_suffix_ids[:, selected_idx],
            )
            selected_suffix_valid_mask[:, selected_idx] = valid_mask
            dense_suffix_ids[:, rel_pos] = torch.where(
                valid_mask,
                gathered_ids,
                dense_suffix_ids[:, rel_pos],
            )

    return (
        dense_suffix_ids,
        dense_suffix_token_mask,
        selected_suffix_ids,
        selected_suffix_valid_mask,
        selected_positions_tensor,
    )


def _extract_left_padded_dense_suffix_relative_ids(suffix_ids, suffix_attention_mask, relative_positions, mask_id):
    if suffix_ids is None or not relative_positions:
        return None, None

    batch_size, suffix_padded_len = suffix_ids.shape
    dense_len = max(int(pos) for pos in relative_positions) + 1
    if dense_len <= 0:
        empty_ids = torch.empty((batch_size, 0), dtype=suffix_ids.dtype, device=suffix_ids.device)
        empty_mask = torch.empty((batch_size, 0), dtype=torch.long, device=suffix_ids.device)
        return empty_ids, empty_mask

    device = suffix_ids.device
    dense_ids = torch.full((batch_size, dense_len), int(mask_id), dtype=suffix_ids.dtype, device=device)
    dense_mask = torch.ones((batch_size, dense_len), dtype=torch.long, device=device)

    if suffix_padded_len == 0:
        return dense_ids, dense_mask

    if suffix_attention_mask is None:
        valid_length = torch.full((batch_size,), suffix_padded_len, dtype=torch.long, device=device)
        suffix_attention_mask = torch.ones_like(suffix_ids, dtype=torch.long, device=device)
    else:
        valid_length = suffix_attention_mask.to(torch.long).sum(dim=-1)

    start_idx = (suffix_padded_len - valid_length).unsqueeze(-1)
    for rel_pos in relative_positions:
        rel_pos = int(rel_pos)
        rel_pos_tensor = torch.full((batch_size, 1), rel_pos, dtype=torch.long, device=device)
        gather_idx = (start_idx + rel_pos_tensor).clamp(min=0, max=max(suffix_padded_len - 1, 0))
        selected_ids = torch.gather(suffix_ids, dim=1, index=gather_idx).squeeze(-1)
        within_actual = rel_pos < valid_length
        if torch.any(within_actual):
            dense_ids[:, rel_pos] = torch.where(within_actual, selected_ids, dense_ids[:, rel_pos])
    return dense_ids, dense_mask


def _suffix_relative_positions_to_head_length_or_none(relative_positions):
    if relative_positions is None:
        return None
    normalized = [int(pos) for pos in relative_positions]
    if normalized == list(range(len(normalized))):
        return len(normalized)
    return None


def _is_nested_suffix_relative_positions(value):
    return isinstance(value, (list, tuple)) and bool(value) and isinstance(value[0], (list, tuple))


def _slice_batch_tensor(value, batch_idx):
    if value is None or not torch.is_tensor(value):
        return value
    return value[batch_idx:batch_idx + 1]


def _merge_candidate_confidence_records(candidate_records_per_sample):
    def _pad_and_concat(lhs, rhs):
        if lhs.dim() >= 2 and rhs.dim() >= 2 and lhs.shape[1] != rhs.shape[1]:
            max_width = max(lhs.shape[1], rhs.shape[1])
            if lhs.shape[1] < max_width:
                pad_shape = list(lhs.shape)
                pad_shape[1] = max_width - lhs.shape[1]
                lhs = torch.cat([lhs, torch.zeros(*pad_shape, dtype=lhs.dtype, device=lhs.device)], dim=1)
            if rhs.shape[1] < max_width:
                pad_shape = list(rhs.shape)
                pad_shape[1] = max_width - rhs.shape[1]
                rhs = torch.cat([rhs, torch.zeros(*pad_shape, dtype=rhs.dtype, device=rhs.device)], dim=1)
        return torch.cat([lhs, rhs], dim=0)

    if not candidate_records_per_sample:
        return []
    merged = {}
    slot_order = []
    for sample_records in candidate_records_per_sample:
        for record in sample_records:
            slot_name = record["slot_name"]
            if slot_name not in merged:
                merged[slot_name] = {key: value for key, value in record.items()}
                slot_order.append(slot_name)
                continue
            target = merged[slot_name]
            for key, value in record.items():
                if torch.is_tensor(value) and torch.is_tensor(target.get(key)):
                    target[key] = _pad_and_concat(target[key], value)
    return [merged[slot_name] for slot_name in slot_order]


def _build_parallel_candidate_position_ids(prefix_token_mask, slot_lengths, slot_position_offsets=None):
    batch_size, prefix_len = prefix_token_mask.shape
    device = prefix_token_mask.device
    real_prefix_len = prefix_token_mask.to(torch.long).sum(dim=-1)
    position_pieces = [
        _build_left_padded_segment_position_ids(
            prefix_token_mask,
            start_positions=torch.zeros_like(real_prefix_len),
            dtype=torch.long,
        )
    ]
    if slot_position_offsets is None:
        slot_position_offsets = [
            torch.arange(slot_length, device=device, dtype=torch.long)
            for slot_length in slot_lengths
        ]
    for slot_length, slot_offsets in zip(slot_lengths, slot_position_offsets):
        slot_start = real_prefix_len
        slot_pos = slot_offsets.unsqueeze(0) + slot_start.unsqueeze(-1)
        position_pieces.extend([slot_pos, slot_pos])
    return torch.cat(position_pieces, dim=1)


@ torch.no_grad()
def evaluate_front_candidate_causal_confidence_dense_suffix_shared(
    model,
    x,
    prefix_ids,
    front_mask_lengths,
    suffix_ids,
    attention_mask,
    suffix_attention_mask,
    cfg_scale,
    mask_id,
    logits_eos_inf,
    confidence_eos_eot_inf,
    tokenizer,
    verbose,
    score_basis,
    score_topk,
    suffix_relative_positions,
    full_suffix_context=False,
):
    front_mask_lengths = [int(slot_length) for slot_length in front_mask_lengths]
    selected_positions = [int(pos) for pos in suffix_relative_positions if int(pos) >= 0]
    if not selected_positions:
        return []

    import torch.nn.functional as F

    prefix_len = prefix_ids.shape[1]
    batch_size = x.shape[0]
    selected_suffix_count = len(selected_positions)
    active_slots = [(slot_idx, slot_length) for slot_idx, slot_length in enumerate(front_mask_lengths) if slot_length > 0]

    mask_dtype = (
        attention_mask.dtype
        if attention_mask is not None
        else (suffix_attention_mask.dtype if suffix_attention_mask is not None else torch.long)
    )
    prefix_token_mask = (
        attention_mask
        if attention_mask is not None
        else torch.ones((batch_size, prefix_len), dtype=mask_dtype, device=model.device)
    )

    (
        dense_suffix_ids,
        dense_suffix_token_mask,
        selected_suffix_ids,
        selected_suffix_valid_mask,
        selected_positions_tensor,
    ) = _build_visible_suffix_window(
        suffix_ids=suffix_ids,
        suffix_attention_mask=suffix_attention_mask,
        selected_positions=selected_positions,
        mask_id=mask_id,
        batch_size=batch_size,
        token_dtype=prefix_ids.dtype,
        mask_dtype=prefix_token_mask.dtype,
        device=model.device,
        full_suffix_context=full_suffix_context,
    )
    dense_suffix_len = int(dense_suffix_ids.shape[1])

    prefix_pos = _build_left_padded_segment_position_ids(
        prefix_token_mask,
        start_positions=torch.zeros(batch_size, dtype=torch.long, device=model.device),
        dtype=torch.long,
    )
    real_prefix_len = prefix_token_mask.to(torch.long).sum(dim=-1)

    seq_len = prefix_len
    slot_layouts = []
    for slot_idx, slot_length in active_slots:
        text_len = slot_length + dense_suffix_len
        mask_len = slot_length + selected_suffix_count
        slot_layouts.append(
            {
                "slot_idx": slot_idx,
                "slot_length": slot_length,
                "text_len": text_len,
                "mask_len": mask_len,
            }
        )
        seq_len += text_len + mask_len

    eval_x = torch.full((batch_size, seq_len), int(mask_id), dtype=torch.long, device=model.device)
    if prefix_len > 0:
        eval_x[:, :prefix_len] = prefix_ids.clone()

    curr_attention_mask = torch.ones((batch_size, seq_len), dtype=prefix_token_mask.dtype, device=model.device)
    if prefix_len > 0:
        curr_attention_mask[:, :prefix_len] = prefix_token_mask

    curr_position_ids = torch.zeros((batch_size, seq_len), dtype=torch.long, device=model.device)
    if prefix_len > 0:
        curr_position_ids[:, :prefix_len] = prefix_pos

    allow = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=model.device)
    if prefix_len > 0:
        allow[:prefix_len, :prefix_len] = True

    candidate_confidence_records = []
    read_cursor = 0
    write_cursor = prefix_len
    for layout in slot_layouts:
        slot_idx = layout["slot_idx"]
        slot_length = layout["slot_length"]
        text_len = layout["text_len"]
        mask_len = layout["mask_len"]

        text_start = write_cursor
        text_end = text_start + text_len
        mask_start = text_end
        mask_end = mask_start + mask_len
        candidate_text_start = text_start
        candidate_text_end = candidate_text_start + slot_length
        dense_suffix_text_start = candidate_text_end

        candidate_ids = x[:, read_cursor:read_cursor + slot_length].to(dtype=eval_x.dtype)
        eval_x[:, candidate_text_start:candidate_text_end] = candidate_ids
        eval_x[:, dense_suffix_text_start:text_end] = dense_suffix_ids

        slot_text_position_ids = torch.cat(
            [
                torch.arange(slot_length, device=model.device, dtype=torch.long).unsqueeze(0) + real_prefix_len.unsqueeze(-1),
                torch.arange(dense_suffix_len, device=model.device, dtype=torch.long).unsqueeze(0)
                + (real_prefix_len + slot_length).unsqueeze(-1),
            ],
            dim=-1,
        )
        slot_mask_position_ids = torch.cat(
            [
                torch.arange(slot_length, device=model.device, dtype=torch.long).unsqueeze(0) + real_prefix_len.unsqueeze(-1),
                selected_positions_tensor.unsqueeze(0) + (real_prefix_len + slot_length).unsqueeze(-1),
            ],
            dim=-1,
        )
        curr_position_ids[:, text_start:text_end] = slot_text_position_ids
        curr_position_ids[:, mask_start:mask_end] = slot_mask_position_ids

        if prefix_len > 0:
            allow[text_start:text_end, :prefix_len] = True
            allow[mask_start:mask_end, :prefix_len] = True

        allow[text_start:text_end, text_start:text_end] = torch.tril(
            torch.ones((text_len, text_len), dtype=torch.bool, device=model.device)
        )
        if slot_length > 0:
            allow[mask_start:mask_start + slot_length, candidate_text_start:candidate_text_end] = torch.tril(
                torch.ones((slot_length, slot_length), dtype=torch.bool, device=model.device),
                diagonal=-1,
            )
        if selected_suffix_count > 0:
            allow[mask_start + slot_length:mask_end, candidate_text_start:candidate_text_end] = True
            for selected_idx, rel_pos in enumerate(selected_positions):
                if rel_pos > 0:
                    allow[
                        mask_start + slot_length + selected_idx,
                        dense_suffix_text_start:dense_suffix_text_start + rel_pos,
                    ] = True

        target_ids = torch.cat([candidate_ids, selected_suffix_ids], dim=-1)
        valid_mask = torch.cat(
            [
                torch.ones((batch_size, slot_length), dtype=torch.bool, device=model.device),
                selected_suffix_valid_mask,
            ],
            dim=-1,
        )
        mask_positions = torch.arange(mask_start, mask_end, device=model.device)
        candidate_confidence_records.append(
            {
                "slot_name": f"front_{slot_idx}",
                "token_ids": target_ids,
                "mask_positions": mask_positions,
                "valid_mask": valid_mask,
                "candidate_token_count": slot_length,
                "suffix_head_length": dense_suffix_len,
            }
        )

        read_cursor += slot_length
        write_cursor = mask_end

    attention_bias = torch.full(
        (seq_len, seq_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=model.device,
    )
    attention_bias.masked_fill_(allow, 0.0)

    prompt_index = (eval_x != int(mask_id))
    prompt_index = prompt_index & curr_attention_mask.bool()
    logits = get_logits(
        model,
        eval_x,
        curr_attention_mask,
        prompt_index,
        cfg_scale,
        int(mask_id),
        position_ids=curr_position_ids,
        attention_bias=attention_bias,
    )
    if logits_eos_inf:
        logits[:, :, 126081] = -torch.inf
    if confidence_eos_eot_inf:
        logits[:, :, 126081] = logits[:, :, 126348] = -torch.inf

    probs = F.softmax(logits.float(), dim=-1)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    for record in candidate_confidence_records:
        valid_mask = record["valid_mask"]
        valid_mask_f = valid_mask.to(probs.dtype)
        valid_count = valid_mask_f.sum(dim=-1).clamp(min=1.0)
        token_ids = record["token_ids"].to(device=probs.device, dtype=torch.long)
        token_conf = torch.gather(
            probs[:, record["mask_positions"]],
            dim=-1,
            index=token_ids.unsqueeze(-1),
        ).squeeze(-1)
        token_logprob = torch.gather(
            logprobs[:, record["mask_positions"]],
            dim=-1,
            index=token_ids.unsqueeze(-1),
        ).squeeze(-1)
        if score_basis == "target_logprob":
            token_score = token_logprob
        elif score_basis == "predictive_entropy":
            token_score = (probs[:, record["mask_positions"]] * logprobs[:, record["mask_positions"]]).sum(dim=-1)
        elif score_basis == "topk_mass":
            k = min(int(score_topk), logprobs.shape[-1])
            if k <= 0:
                raise ValueError("score_topk must be positive when score_basis='topk_mass'")
            token_score = torch.topk(probs[:, record["mask_positions"]], k=k, dim=-1).values.sum(dim=-1)
        elif score_basis == "topk_logprob_sum":
            k = min(int(score_topk), logprobs.shape[-1])
            if k <= 0:
                raise ValueError("score_topk must be positive when score_basis='topk_logprob_sum'")
            token_score = torch.topk(logprobs[:, record["mask_positions"]], k=k, dim=-1).values.sum(dim=-1)
        elif score_basis == "topk_entropy":
            k = min(int(score_topk), probs.shape[-1])
            if k <= 0:
                raise ValueError("score_topk must be positive when score_basis='topk_entropy'")
            topk_probs = torch.topk(probs[:, record["mask_positions"]], k=k, dim=-1).values
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(topk_probs.dtype).tiny)
            token_score = (topk_probs * topk_probs.clamp_min(torch.finfo(topk_probs.dtype).tiny).log()).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported score_basis: {score_basis}")
        record["token_conf"] = token_conf
        record["token_logprob"] = token_logprob
        record["avg_conf"] = (token_conf * valid_mask_f).sum(dim=-1) / valid_count
        record["min_conf"] = torch.where(
            valid_mask,
            token_conf,
            torch.full_like(token_conf, torch.inf),
        ).min(dim=-1).values
        record["token_score"] = token_score
        record["avg_score"] = (token_score * valid_mask_f).sum(dim=-1) / valid_count
        record["min_score"] = torch.where(
            valid_mask,
            token_score,
            torch.full_like(token_score, torch.inf),
        ).min(dim=-1).values
        record["score_basis"] = score_basis
        del record["mask_positions"]

    return candidate_confidence_records


@ torch.no_grad()
def evaluate_front_candidate_causal_confidence(model, x, prefix_ids, front_mask_lengths, suffix_ids=None, attention_mask=None, suffix_attention_mask=None,
                                               cfg_scale=0., mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
                                               position_mode='max', tokenizer=None, verbose=True,
                                               score_basis='target_logprob', score_topk=32, suffix_head_length=0,
                                               suffix_relative_positions=None, full_suffix_context=False):
    front_mask_lengths = [int(slot_length) for slot_length in front_mask_lengths]
    prefix_len = prefix_ids.shape[1]
    batch_size = x.shape[0]
    total_front_len = sum(front_mask_lengths)
    active_slots = [(slot_idx, slot_length) for slot_idx, slot_length in enumerate(front_mask_lengths) if slot_length > 0]
    if _is_nested_suffix_relative_positions(suffix_relative_positions):
        candidate_records_per_sample = []
        for batch_idx, sample_suffix_positions in enumerate(suffix_relative_positions):
            sample_records = evaluate_front_candidate_causal_confidence(
                model=model,
                x=x[batch_idx:batch_idx + 1],
                prefix_ids=prefix_ids[batch_idx:batch_idx + 1],
                front_mask_lengths=front_mask_lengths,
                suffix_ids=_slice_batch_tensor(suffix_ids, batch_idx),
                attention_mask=_slice_batch_tensor(attention_mask, batch_idx),
                suffix_attention_mask=_slice_batch_tensor(suffix_attention_mask, batch_idx),
                cfg_scale=cfg_scale,
                mask_id=mask_id,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
                position_mode=position_mode,
                tokenizer=tokenizer,
                verbose=False,
                score_basis=score_basis,
                score_topk=score_topk,
                suffix_head_length=suffix_head_length,
                suffix_relative_positions=list(sample_suffix_positions),
                full_suffix_context=full_suffix_context,
            )
            candidate_records_per_sample.append(sample_records)
        return _merge_candidate_confidence_records(candidate_records_per_sample)
    if suffix_relative_positions is not None:
        suffix_relative_positions = [int(pos) for pos in suffix_relative_positions if int(pos) >= 0]
        if not full_suffix_context:
            contiguous_head_length = _suffix_relative_positions_to_head_length_or_none(suffix_relative_positions)
            if contiguous_head_length is not None:
                suffix_head_length = contiguous_head_length
                suffix_relative_positions = None
    if full_suffix_context and suffix_relative_positions is None:
        suffix_relative_positions = list(range(max(int(0 if suffix_head_length is None else suffix_head_length), 0)))
    if suffix_relative_positions:
        return evaluate_front_candidate_causal_confidence_dense_suffix_shared(
            model=model,
            x=x,
            prefix_ids=prefix_ids,
            front_mask_lengths=front_mask_lengths,
            suffix_ids=suffix_ids,
            attention_mask=attention_mask,
            suffix_attention_mask=suffix_attention_mask,
            cfg_scale=cfg_scale,
            mask_id=mask_id,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
            tokenizer=tokenizer,
            verbose=verbose,
            score_basis=score_basis,
            score_topk=score_topk,
            suffix_relative_positions=suffix_relative_positions,
            full_suffix_context=full_suffix_context,
        )
    requested_suffix_head_length = max(int(0 if suffix_head_length is None else suffix_head_length), 0)
    available_suffix_head_length = suffix_ids.shape[1] if suffix_ids is not None else 0
    suffix_head_length = min(requested_suffix_head_length, available_suffix_head_length)

    mask_dtype = (
        attention_mask.dtype
        if attention_mask is not None
        else (suffix_attention_mask.dtype if suffix_attention_mask is not None else torch.long)
    )
    prefix_token_mask = (
        attention_mask
        if attention_mask is not None
        else torch.ones((batch_size, prefix_len), dtype=mask_dtype, device=model.device)
    )
    suffix_head_ids = None
    suffix_head_token_mask = None
    if suffix_relative_positions is not None:
        suffix_head_ids, suffix_head_token_mask = _extract_left_padded_dense_suffix_relative_ids(
            suffix_ids=suffix_ids,
            suffix_attention_mask=suffix_attention_mask,
            relative_positions=suffix_relative_positions,
            mask_id=mask_id,
        )
        if suffix_head_token_mask is None:
            suffix_head_token_mask = torch.ones(
                (batch_size, 0),
                dtype=mask_dtype,
                device=model.device,
            )
        else:
            suffix_head_token_mask = suffix_head_token_mask.to(device=model.device, dtype=mask_dtype)
        suffix_head_length = int(suffix_head_ids.shape[1]) if suffix_head_ids is not None else 0
    elif suffix_head_length > 0:
        suffix_head_ids, suffix_head_token_mask = _extract_left_padded_suffix_head_ids(
            suffix_ids=suffix_ids,
            suffix_attention_mask=suffix_attention_mask,
            head_length=suffix_head_length,
        )
        if suffix_head_token_mask is None:
            suffix_head_token_mask = torch.ones((batch_size, suffix_head_length), dtype=mask_dtype, device=model.device)
        else:
            suffix_head_token_mask = suffix_head_token_mask.to(device=model.device, dtype=mask_dtype)

    slot_eval_lengths = []
    slot_token_mask_pieces = []
    for _, slot_length in active_slots:
        candidate_token_mask = torch.ones((batch_size, slot_length), dtype=mask_dtype, device=model.device)
        if suffix_head_length > 0:
            slot_token_mask = torch.cat([candidate_token_mask, suffix_head_token_mask], dim=-1)
        else:
            slot_token_mask = candidate_token_mask
        slot_eval_lengths.append(slot_token_mask.shape[1])
        slot_token_mask_pieces.append(slot_token_mask)
    duplicated_slot_masks = []
    for slot_mask in slot_token_mask_pieces:
        duplicated_slot_masks.extend([slot_mask, slot_mask])
    curr_attention_mask = torch.cat(
        [prefix_token_mask] + duplicated_slot_masks,
        dim=-1,
    )

    curr_attention_bias = _build_parallel_candidate_causal_attention_bias(
        prefix_len=prefix_len,
        slot_lengths=slot_eval_lengths,
        device=model.device,
    )
    curr_position_ids = _build_parallel_candidate_position_ids(
        prefix_token_mask=prefix_token_mask,
        slot_lengths=slot_eval_lengths,
    )

    eval_seq_len = prefix_len + sum(slot_length * 2 for slot_length in slot_eval_lengths)
    eval_x = torch.full((batch_size, eval_seq_len), mask_id, dtype=torch.long, device=model.device)
    if prefix_len > 0:
        eval_x[:, :prefix_len] = prefix_ids.clone()

    candidate_confidence_records = []
    read_cursor = 0
    write_cursor = prefix_len
    for record_idx, (slot_idx, slot_length) in enumerate(active_slots):
        slot_eval_length = slot_eval_lengths[record_idx]
        source_positions = torch.arange(read_cursor, read_cursor + slot_length, device=model.device)
        text_positions = torch.arange(write_cursor, write_cursor + slot_eval_length, device=model.device)
        mask_positions = torch.arange(write_cursor + slot_eval_length, write_cursor + 2 * slot_eval_length, device=model.device)
        candidate_token_ids = x[:, source_positions]
        token_ids = candidate_token_ids
        token_valid_mask = torch.ones((batch_size, slot_length), dtype=torch.bool, device=model.device)
        if suffix_head_length > 0:
            token_ids = torch.cat([token_ids, suffix_head_ids], dim=-1)
            token_valid_mask = torch.cat([token_valid_mask, suffix_head_token_mask.bool()], dim=-1)
        token_ids = token_ids.to(dtype=eval_x.dtype)
        visible_token_ids = torch.where(
            token_valid_mask,
            token_ids,
            torch.full_like(token_ids, mask_id),
        )
        eval_x[:, text_positions] = visible_token_ids
        candidate_confidence_records.append(
            {
                "slot_name": f"front_{slot_idx}",
                "token_ids": token_ids,
                "mask_positions": mask_positions,
                "valid_mask": token_valid_mask,
                "candidate_token_count": slot_length,
                "suffix_head_length": suffix_head_length,
            }
        )
        read_cursor += slot_length
        write_cursor += 2 * slot_eval_length

    prompt_index = (eval_x != mask_id)
    if curr_attention_mask is not None:
        prompt_index = prompt_index & curr_attention_mask.bool()
    logits = get_logits(
        model,
        eval_x,
        curr_attention_mask,
        prompt_index,
        cfg_scale,
        mask_id,
        position_ids=curr_position_ids,
        attention_bias=curr_attention_bias,
    )
    if logits_eos_inf:
        logits[:, :, 126081] = -torch.inf
    if confidence_eos_eot_inf:
        logits[:, :, 126081] = logits[:, :, 126348] = -torch.inf

    probs = F.softmax(logits.float(), dim=-1)
    logprobs = F.log_softmax(logits.float(), dim=-1)
    for record in candidate_confidence_records:
        debug_mask_positions = record["mask_positions"].clone()
        valid_mask = record["valid_mask"]
        valid_mask_f = valid_mask.to(probs.dtype)
        valid_count = valid_mask_f.sum(dim=-1).clamp(min=1.0)
        token_ids = record["token_ids"].to(device=probs.device, dtype=torch.long)
        token_conf = torch.gather(
            probs[:, record["mask_positions"]],
            dim=-1,
            index=token_ids.unsqueeze(-1),
        ).squeeze(-1)
        token_logprob = torch.gather(
            logprobs[:, record["mask_positions"]],
            dim=-1,
            index=token_ids.unsqueeze(-1),
        ).squeeze(-1)
        if score_basis == "target_logprob":
            token_score = token_logprob
        elif score_basis == "predictive_entropy":
            token_score = (probs[:, record["mask_positions"]] * logprobs[:, record["mask_positions"]]).sum(dim=-1)
        elif score_basis == "topk_mass":
            k = min(int(score_topk), logprobs.shape[-1])
            if k <= 0:
                raise ValueError("score_topk must be positive when score_basis='topk_mass'")
            token_score = torch.topk(probs[:, record["mask_positions"]], k=k, dim=-1).values.sum(dim=-1)
        elif score_basis == "topk_logprob_sum":
            k = min(int(score_topk), logprobs.shape[-1])
            if k <= 0:
                raise ValueError("score_topk must be positive when score_basis='topk_logprob_sum'")
            token_score = torch.topk(logprobs[:, record["mask_positions"]], k=k, dim=-1).values.sum(dim=-1)
        elif score_basis == "topk_entropy":
            k = min(int(score_topk), probs.shape[-1])
            if k <= 0:
                raise ValueError("score_topk must be positive when score_basis='topk_entropy'")
            topk_probs = torch.topk(probs[:, record["mask_positions"]], k=k, dim=-1).values
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(torch.finfo(topk_probs.dtype).tiny)
            token_score = (topk_probs * topk_probs.clamp_min(torch.finfo(topk_probs.dtype).tiny).log()).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported score_basis: {score_basis}")
        record["token_conf"] = token_conf
        record["token_logprob"] = token_logprob
        record["avg_conf"] = (token_conf * valid_mask_f).sum(dim=-1) / valid_count
        record["min_conf"] = torch.where(
            valid_mask,
            token_conf,
            torch.full_like(token_conf, torch.inf),
        ).min(dim=-1).values
        record["token_score"] = token_score
        record["avg_score"] = (token_score * valid_mask_f).sum(dim=-1) / valid_count
        record["min_score"] = torch.where(
            valid_mask,
            token_score,
            torch.full_like(token_score, torch.inf),
        ).min(dim=-1).values
        record["score_basis"] = score_basis
        record["debug_second_token_top5"] = None
        if debug_mask_positions.numel() > 1:
            topk_vals, topk_ids = torch.topk(probs[:, debug_mask_positions[1]], k=5, dim=-1)
            record["debug_second_token_top5"] = {
                "ids": topk_ids.detach().cpu(),
                "vals": topk_vals.detach().cpu(),
            }
        del record["mask_positions"]

    if verbose:
        for batch_idx in range(batch_size):
            print(f"[candidate_causal_confidence] sample={batch_idx}")
            sample_avg_confs = []
            for record in candidate_confidence_records:
                token_id_list = record["token_ids"][batch_idx].detach().cpu().tolist()
                token_conf_list = record["token_conf"][batch_idx].detach().cpu().tolist()
                token_valid_list = record["valid_mask"][batch_idx].detach().cpu().tolist()
                avg_conf = record["avg_conf"][batch_idx].item()
                min_conf = record["min_conf"][batch_idx].item()
                avg_score = record["avg_score"][batch_idx].item()
                min_score = record["min_score"][batch_idx].item()
                sample_avg_confs.append(avg_conf)
                token_str_list = (
                    [None] * len(token_id_list)
                    if tokenizer is None
                    else tokenizer.convert_ids_to_tokens(token_id_list)
                )
                print(
                    f"### slot={record['slot_name']} avg_conf={avg_conf:.6f} min_conf={min_conf:.6f} "
                    f"avg_score={avg_score:.6f} min_score={min_score:.6f} score_basis={record['score_basis']}"
                )
                print(
                    "    "
                    + " ".join(
                        f"[{token_str}]{token_conf:.6f}"
                        for token_str, token_conf, token_valid in zip(
                            token_str_list,
                            token_conf_list,
                            token_valid_list,
                        )
                        if token_valid
                    )
                )
                if tokenizer is not None and record["debug_second_token_top5"] is not None:
                    topk_id_list = record["debug_second_token_top5"]["ids"][batch_idx].tolist()
                    topk_val_list = record["debug_second_token_top5"]["vals"][batch_idx].tolist()
                    topk_token_list = tokenizer.convert_ids_to_tokens(topk_id_list)
                    print(
                        "    second_token_top5="
                        + " ".join(
                            f"{token_id}:{token_str}:{score:.6f}"
                            for token_id, token_str, score in zip(topk_id_list, topk_token_list, topk_val_list)
                        )
                    )
            if sample_avg_confs:
                print(f"  overall_avg_conf={sum(sample_avg_confs) / len(sample_avg_confs):.6f}")

    for record in candidate_confidence_records:
        del record["debug_second_token_top5"]

    return candidate_confidence_records


def get_logits(model, x, curr_attention_mask, prompt_index, cfg_scale, mask_id, position_ids=None, attention_bias=None):
    """Runs the model once, optionally with classifier-free guidance."""
    if cfg_scale > 0.:
        un_x = x.clone()
        un_x[prompt_index] = mask_id
        x_ = torch.cat([x, un_x], dim=0)
        a_mask_ = torch.cat([curr_attention_mask, curr_attention_mask], dim=0) if curr_attention_mask is not None else None
        # print(a_mask_)
        pos_ids_ = (
            torch.cat([position_ids, position_ids], dim=0) if position_ids is not None else None
        )
        if attention_bias is None:
            attn_bias_ = None
        elif attention_bias.dim() == 2:
            attn_bias_ = attention_bias
        else:
            attn_bias_ = torch.cat([attention_bias, attention_bias], dim=0)
        # print(pos_ids_)
        # exit(0)
        logits = _call_model_logits(
            model,
            x_,
            attention_mask=a_mask_,
            position_ids=pos_ids_,
            attention_bias=attn_bias_,
        )
        logits, un_logits = torch.chunk(logits, 2, dim=0)
        logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
        return logits
    return _call_model_logits(
        model,
        x,
        attention_mask=curr_attention_mask,
        position_ids=position_ids,
        attention_bias=attention_bias,
    )


@torch.no_grad()
def collect_generate_mask_first_pass_suffix_attention_scores(
    model,
    prefix_ids,
    front_mask_lengths,
    suffix_ids=None,
    attention_mask=None,
    suffix_attention_mask=None,
    gen_length=8,
    cfg_scale=0.0,
    mask_id=126336,
    position_mode="max",
):
    front_mask_lengths = [int(slot_length) for slot_length in front_mask_lengths]
    prefix_len = prefix_ids.shape[1]
    suffix_len = suffix_ids.shape[1] if suffix_ids is not None else 0
    total_front_len = sum(front_mask_lengths)
    batch_size = prefix_ids.shape[0]
    seq_len = total_front_len + prefix_len + int(gen_length) + suffix_len

    if seq_len <= 0 or int(gen_length) <= 0 or suffix_len <= 0:
        return None

    mask_dtype = (
        attention_mask.dtype
        if attention_mask is not None
        else (suffix_attention_mask.dtype if suffix_attention_mask is not None else torch.long)
    )
    prefix_token_mask = (
        attention_mask
        if attention_mask is not None
        else torch.ones((batch_size, prefix_len), dtype=mask_dtype, device=model.device)
    )
    suffix_token_mask = (
        suffix_attention_mask
        if suffix_attention_mask is not None
        else torch.ones((batch_size, suffix_len), dtype=mask_dtype, device=model.device)
    )

    prefix_start = total_front_len
    prefix_end = prefix_start + prefix_len
    middle_start = prefix_end
    middle_end = middle_start + int(gen_length)
    suffix_start = middle_end

    x = torch.full((batch_size, seq_len), int(mask_id), dtype=torch.long, device=model.device)
    if prefix_len > 0:
        x[:, prefix_start:prefix_end] = prefix_ids.clone()
    if suffix_ids is not None:
        x[:, suffix_start:] = suffix_ids.clone()

    attention_mask_pieces = []
    for slot_length in front_mask_lengths:
        if slot_length > 0:
            attention_mask_pieces.append(
                torch.ones((batch_size, slot_length), dtype=prefix_token_mask.dtype, device=model.device)
            )
    attention_mask_pieces.extend(
        [
            prefix_token_mask,
            torch.ones((batch_size, int(gen_length)), dtype=prefix_token_mask.dtype, device=model.device),
            suffix_token_mask,
        ]
    )
    curr_attention_mask = torch.cat(attention_mask_pieces, dim=-1)

    curr_position_ids = _build_mask_slot_position_ids(
        batch_size=batch_size,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        front_mask_lengths=front_mask_lengths,
        middle_mask_length=int(gen_length),
        device=model.device,
        position_mode=position_mode,
        prefix_token_mask=prefix_token_mask,
        suffix_token_mask=suffix_token_mask,
    )

    allow = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=model.device)
    allow[prefix_start:seq_len, prefix_start:seq_len] = True
    cursor = 0
    for slot_length in front_mask_lengths:
        if slot_length <= 0:
            continue
        slot_start_idx = cursor
        slot_end_idx = slot_start_idx + slot_length
        allow[slot_start_idx:slot_end_idx, slot_start_idx:slot_end_idx] = True
        if prefix_len > 0:
            allow[slot_start_idx:slot_end_idx, prefix_start:prefix_end] = True
        if suffix_start < seq_len:
            allow[slot_start_idx:slot_end_idx, suffix_start:seq_len] = True
        cursor = slot_end_idx

    attention_bias = torch.full(
        (seq_len, seq_len),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=model.device,
    )
    attention_bias.masked_fill_(allow, 0.0)

    prompt_index = (x != int(mask_id)) & curr_attention_mask.bool()
    get_logits(
        model,
        x,
        curr_attention_mask,
        prompt_index,
        cfg_scale,
        int(mask_id),
        position_ids=curr_position_ids,
        attention_bias=attention_bias,
    )
    return {
        "middle_positions": list(range(middle_start, middle_end)),
        "suffix_positions": list(range(suffix_start, suffix_start + suffix_len)),
    }


def _build_mask_slot_position_ids(batch_size, prefix_len, suffix_len, front_mask_lengths, middle_mask_length, device, position_mode="max",
                                  prefix_token_mask=None, suffix_token_mask=None):
    if position_mode not in {"max", "max_left", "interpolate", "round"}:
        raise ValueError(f"Unsupported position_mode: {position_mode}")

    if prefix_token_mask is None:
        prefix_token_mask = torch.ones((batch_size, prefix_len), device=device, dtype=torch.long)
    else:
        prefix_token_mask = prefix_token_mask.to(device=device)
    if suffix_token_mask is None:
        suffix_token_mask = torch.ones((batch_size, suffix_len), device=device, dtype=torch.long)
    else:
        suffix_token_mask = suffix_token_mask.to(device=device)

    real_prefix_len = prefix_token_mask.to(torch.long).sum(dim=-1)

    if position_mode in {"max", "max_left"}:
        prefix_pos = _build_left_padded_segment_position_ids(
            prefix_token_mask,
            start_positions=torch.zeros_like(real_prefix_len),
            dtype=torch.long,
        )
        all_mask_lengths = front_mask_lengths + [middle_mask_length]
        max_mask_length = max(all_mask_lengths) if all_mask_lengths else 0

        slot_pos_list = []
        for slot_length in front_mask_lengths:
            slot_start = real_prefix_len
            if position_mode == "max_left":
                slot_start = real_prefix_len + max_mask_length - slot_length
            slot_pos_list.append(
                torch.arange(slot_length, device=device, dtype=torch.long).unsqueeze(0) + slot_start.unsqueeze(-1)
            )

        middle_start = real_prefix_len
        if position_mode == "max_left":
            middle_start = real_prefix_len + max_mask_length - middle_mask_length
        middle_pos = (
            torch.arange(middle_mask_length, device=device, dtype=torch.long).unsqueeze(0)
            + middle_start.unsqueeze(-1)
        )
        suffix_pos = _build_left_padded_segment_position_ids(
            suffix_token_mask,
            start_positions=real_prefix_len + max_mask_length,
            dtype=torch.long,
        )
    else:
        prefix_pos = _build_left_padded_segment_position_ids(
            prefix_token_mask,
            start_positions=torch.zeros_like(real_prefix_len),
            dtype=torch.float,
        )

        slot_pos_list = []
        interp_start = real_prefix_len.to(torch.float)
        interp_end = real_prefix_len.to(torch.float) + float(max(middle_mask_length - 1, 0))
        for slot_length in front_mask_lengths:
            if slot_length == 0:
                slot_pos_list.append(torch.empty((batch_size, 0), device=device, dtype=torch.float))
            elif slot_length == 1:
                slot_pos_list.append(interp_start.unsqueeze(-1))
            else:
                interp_weight = torch.linspace(0.0, 1.0, steps=slot_length, device=device, dtype=torch.float).unsqueeze(0)
                slot_pos_list.append(
                    interp_start.unsqueeze(-1) + interp_weight * (interp_end - interp_start).unsqueeze(-1)
                )

        middle_pos = (
            torch.arange(middle_mask_length, device=device, dtype=torch.float).unsqueeze(0)
            + real_prefix_len.to(torch.float).unsqueeze(-1)
        )
        suffix_pos = _build_left_padded_segment_position_ids(
            suffix_token_mask,
            start_positions=real_prefix_len + middle_mask_length,
            dtype=torch.float,
        )

        if position_mode == "round":
            slot_pos_list = [torch.round(slot_pos).to(dtype=torch.long) for slot_pos in slot_pos_list]
            prefix_pos = torch.round(prefix_pos).to(dtype=torch.long)
            middle_pos = torch.round(middle_pos).to(dtype=torch.long)
            suffix_pos = torch.round(suffix_pos).to(dtype=torch.long)

    curr_position_ids = torch.cat(slot_pos_list + [prefix_pos, middle_pos, suffix_pos], dim=1)
    return curr_position_ids


def _suppress_logits_token_ids(logits, token_ids):
    if token_ids:
        logits[:, :, [int(token_id) for token_id in token_ids]] = -torch.inf
    return logits


@ torch.no_grad()
def generate_mask(model, prefix_ids, front_mask_lengths, suffix_ids=None, attention_mask=None, suffix_attention_mask=None, steps=None, gen_length=8, block_length=None, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
             position_mode='max',
             return_confidence=False, tokenizer=None, suppress_token_ids=None):
    """
    Multi-slot masked generation with layout:
    [front slot 1]...[front slot n][prefix][middle mask slot][suffix]

    Attention rules:
    - [prefix][middle mask][suffix] are fully bidirectional.
    - Each front slot attends only to itself plus prefix/suffix.
    - Core tokens do not attend to any front slot.

    Decoding rule:
    - Run one forward pass per step over the whole sequence.
    - Each active slot selects exactly one masked position independently per step.

    Position rule:
    - "max": keep the current max-length based layout.
    - "max_left": still use the longest-slot window, but right-align each mask
      slot inside that window so the ghost gap is on the left.
    - "interpolate": fix the core [prefix][middle][suffix] by gen_length, and
      interpolate each front slot inside the middle span.
    - "round": interpolate first, then round the positions to integers.

    """
    front_mask_lengths = [int(slot_length) for slot_length in front_mask_lengths]
    if any(slot_length < 0 for slot_length in front_mask_lengths):
        raise ValueError("front_mask_lengths must be non-negative")

    prefix_len = prefix_ids.shape[1]
    suffix_len = suffix_ids.shape[1] if suffix_ids is not None else 0
    total_front_len = sum(front_mask_lengths)
    curr_gen_length = gen_length
    total_mask_len = total_front_len + curr_gen_length
    slot_lengths = [slot_length for slot_length in front_mask_lengths if slot_length > 0]
    if curr_gen_length > 0:
        slot_lengths.append(curr_gen_length)

    if total_mask_len == 0:
        raise ValueError("generate_mask requires at least one masked token")
    curr_steps = steps if steps is not None else max(slot_lengths)
    if curr_steps <= 0:
        raise ValueError("steps must be positive")

    batch_size = prefix_ids.shape[0]
    seq_len = total_front_len + prefix_len + curr_gen_length + suffix_len

    curr_attention_mask = None
    prefix_token_mask = None
    suffix_token_mask = None
    if attention_mask is not None or suffix_attention_mask is not None:
        mask_dtype = attention_mask.dtype if attention_mask is not None else suffix_attention_mask.dtype
        prefix_token_mask = (
            attention_mask
            if attention_mask is not None
            else torch.ones((batch_size, prefix_len), dtype=mask_dtype, device=model.device)
        )
        suffix_token_mask = (
            suffix_attention_mask
            if suffix_attention_mask is not None
            else torch.ones((batch_size, suffix_len), dtype=mask_dtype, device=model.device)
        )

    prefix_start = total_front_len
    prefix_end = prefix_start + prefix_len
    middle_start = prefix_end
    middle_end = middle_start + curr_gen_length
    suffix_start = middle_end

    x = torch.full((batch_size, seq_len), mask_id, dtype=torch.long, device=model.device)

    slot_specs = []
    mask_positions = []
    cursor = 0
    attention_mask_pieces = [] if (prefix_token_mask is not None or suffix_token_mask is not None) else None
    for slot_idx, slot_length in enumerate(front_mask_lengths):
        if slot_length == 0:
            continue
        slot_end = cursor + slot_length
        slot_positions = torch.arange(cursor, slot_end, device=model.device)
        if attention_mask_pieces is not None:
            attention_mask_pieces.append(
                torch.ones((batch_size, slot_length), dtype=prefix_token_mask.dtype, device=model.device)
            )
        cursor = slot_end

        slot_specs.append(
            {
                "name": f"front_{slot_idx}",
                "positions": slot_positions,
                "start": int(slot_positions[0].item()),
                "end": int(slot_positions[-1].item()) + 1,
            }
        )
        mask_positions.append(slot_positions)
    x[:, prefix_start:prefix_end] = prefix_ids.clone()
    if attention_mask_pieces is not None:
        attention_mask_pieces.append(prefix_token_mask)
    if curr_gen_length > 0:
        middle_positions = torch.arange(middle_start, middle_end, device=model.device)
        slot_specs.append(
            {
                "name": "middle",
                "positions": middle_positions,
                "start": int(middle_positions[0].item()),
                "end": int(middle_positions[-1].item()) + 1,
            }
        )
        mask_positions.append(middle_positions)
        if attention_mask_pieces is not None:
            attention_mask_pieces.append(
                torch.ones((batch_size, curr_gen_length), dtype=prefix_token_mask.dtype, device=model.device)
            )
    if suffix_ids is not None:
        x[:, suffix_start:] = suffix_ids.clone()
    if attention_mask_pieces is not None:
        attention_mask_pieces.append(suffix_token_mask)

    mask_positions = torch.cat(mask_positions, dim=0)
    track_confidence = bool(return_confidence)
    mask_token_conf = None
    if track_confidence:
        mask_token_conf = torch.full((batch_size, total_mask_len), float('nan'), device=model.device)

    base_position_ids = _build_mask_slot_position_ids(
        batch_size=batch_size,
        prefix_len=prefix_len,
        suffix_len=suffix_len,
        front_mask_lengths=front_mask_lengths,
        middle_mask_length=curr_gen_length,
        device=model.device,
        position_mode=position_mode,
        prefix_token_mask=prefix_token_mask,
        suffix_token_mask=suffix_token_mask,
    )
    base_slot_position_pieces = []
    base_cursor = 0
    for slot_length in front_mask_lengths:
        if slot_length == 0:
            continue
        base_slot_position_pieces.append(base_position_ids[:, base_cursor:base_cursor + slot_length])
        base_cursor += slot_length
    base_prefix_positions = base_position_ids[:, total_front_len: total_front_len + prefix_len]
    base_middle_positions = base_position_ids[:, total_front_len + prefix_len: total_front_len + prefix_len + curr_gen_length]
    base_suffix_positions = base_position_ids[:, total_front_len + prefix_len + curr_gen_length:]

    curr_position_pieces = []
    base_slot_idx = 0
    for slot_length in front_mask_lengths:
        if slot_length == 0:
            continue
        curr_position_pieces.append(base_slot_position_pieces[base_slot_idx])
        base_slot_idx += 1
    curr_position_pieces.extend([base_prefix_positions, base_middle_positions])
    curr_position_pieces.append(base_suffix_positions)
    curr_position_ids = torch.cat(curr_position_pieces, dim=1)

    def _build_candidate_attention_bias():
        allow = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=model.device)
        allow[prefix_start:seq_len, prefix_start:seq_len] = True

        for slot_spec in slot_specs:
            if slot_spec["name"] == "middle":
                continue
            slot_start_idx = slot_spec["start"]
            slot_end_idx = slot_spec["end"]
            allow[slot_start_idx:slot_end_idx, slot_start_idx:slot_end_idx] = True
            if prefix_len > 0:
                allow[slot_start_idx:slot_end_idx, prefix_start:prefix_end] = True
            if suffix_start < seq_len:
                allow[slot_start_idx:slot_end_idx, suffix_start:seq_len] = True

        attention_bias = torch.full(
            (seq_len, seq_len),
            torch.finfo(torch.float32).min,
            dtype=torch.float32,
            device=model.device,
        )
        attention_bias.masked_fill_(allow, 0.0)
        return attention_bias

    curr_attention_bias = _build_candidate_attention_bias()
    if attention_mask_pieces is not None:
        curr_attention_mask = torch.cat(attention_mask_pieces, dim=-1)
    else:
        curr_attention_mask = None

    prompt_index = (x != mask_id)
    if curr_attention_mask is not None:
        prompt_index = prompt_index & curr_attention_mask.bool()
    for step_idx in range(curr_steps):
        mask_index = (x == mask_id)
        logits = get_logits(
            model,
            x,
            curr_attention_mask,
            prompt_index,
            cfg_scale,
            mask_id,
            position_ids=curr_position_ids,
            attention_bias=curr_attention_bias,
        )

        if logits_eos_inf:
            logits[:, :, 126081] = -torch.inf
        _suppress_logits_token_ids(logits, suppress_token_ids)
        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        if confidence_eos_eot_inf:
            logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

        if remasking == 'low_confidence':
            p = F.softmax(logits, dim=-1)
            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
        elif remasking == 'random':
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
        else:
            raise NotImplementedError(remasking)

        x0 = torch.where(mask_index, x0, x)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
        for batch_idx in range(batch_size):
            for slot_spec in slot_specs:
                slot_positions = slot_spec["positions"]
                slot_mask = mask_index[batch_idx, slot_positions]
                if not torch.any(slot_mask):
                    continue
                candidate_positions = slot_positions[slot_mask]
                candidate_conf = x0_p[batch_idx, candidate_positions]
                selected_local_idx = torch.argmax(candidate_conf)
                selected_pos = candidate_positions[selected_local_idx]
                transfer_index[batch_idx, selected_pos] = True
        if track_confidence:
            transfer_mask_in_slots = transfer_index[:, mask_positions]
            mask_conf_source = x0_p[:, mask_positions].to(mask_token_conf.dtype)
            mask_token_conf[transfer_mask_in_slots] = mask_conf_source[transfer_mask_in_slots]
        x[transfer_index] = x0[transfer_index]

    if track_confidence:
        valid_first_conf_mask = torch.isfinite(mask_token_conf)
        safe_first_conf = torch.where(valid_first_conf_mask, mask_token_conf, torch.zeros_like(mask_token_conf))
        mask_avg_conf = safe_first_conf.sum(dim=-1) / valid_first_conf_mask.sum(dim=-1).clamp(min=1)

        final_logits = get_logits(
            model,
            x,
            curr_attention_mask,
            prompt_index,
            cfg_scale,
            mask_id,
            position_ids=curr_position_ids,
            attention_bias=curr_attention_bias,
        )
        if logits_eos_inf:
            final_logits[:, :, 126081] = -torch.inf
        _suppress_logits_token_ids(final_logits, suppress_token_ids)
        if confidence_eos_eot_inf:
            final_logits[:, :, 126081] = final_logits[:, :, 126348] = -torch.inf
        final_probs = F.softmax(final_logits, dim=-1)
        masked_ids = x[:, mask_positions]
        final_token_conf = torch.gather(
            final_probs[:, mask_positions],
            dim=-1,
            index=masked_ids.unsqueeze(-1),
        ).squeeze(-1)
        final_avg_conf = final_token_conf.mean(dim=-1)

        return x, mask_token_conf, mask_avg_conf, final_token_conf, final_avg_conf
    return x
