"""PILL inference routines extracted from the original implementation."""

import time
import torch
import torch.nn.functional as F
from .models.generation_utils_dream import DreamGenerationConfig
from .selection import (
    run_generation_batch,
    build_posthoc_score_by_slot,
    build_suffix_selection_details,
    candidate_has_suppressed_token,
    candidate_has_valid_python_ast,
    clear_light_attn_collector,
    compute_slot_selection_score,
    get_special_token_suppress_ids,
    install_light_attn_collector,
    merge_suffix_positions,
    passes_token_score_threshold,
    print_suffix_selection_details,
    resolve_attn_layer_index,
    resolve_base_suffix_positions,
    add_stage_timing,
)


def build_dream_parallel_generation_batch(
    tokenizer, problems_batch, candidate_lengths, mask_id: int, device: str
):
    candidate_lengths = [int(length) for length in candidate_lengths]
    if not candidate_lengths:
        raise ValueError(
            "candidate_lengths must be non-empty for Dream parallel slot decoding."
        )
    prompts = [problem["prompt"] for problem in problems_batch]
    suffixes = [problem["suffix"] for problem in problems_batch]
    encoded_prefix = tokenizer(
        prompts, add_special_tokens=False, padding=True, return_tensors="pt"
    )
    encoded_suffix = tokenizer(
        suffixes, add_special_tokens=False, padding=True, return_tensors="pt"
    )
    prefix_ids = encoded_prefix["input_ids"].to(device)
    prefix_mask = encoded_prefix["attention_mask"].to(device)
    suffix_ids = encoded_suffix["input_ids"].to(device)
    suffix_mask = encoded_suffix["attention_mask"].to(device)
    batch_size, prefix_len = prefix_ids.shape
    suffix_len = suffix_ids.shape[1]
    explicit_tail_lengths = [
        max(candidate_length - 1, 0) for candidate_length in candidate_lengths
    ]
    max_input_len = max(
        (
            prefix_len + candidate_length + suffix_len + explicit_tail_length
            for candidate_length, explicit_tail_length in zip(
                candidate_lengths, explicit_tail_lengths
            )
        )
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = torch.full(
        (batch_size * len(candidate_lengths), max_input_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size * len(candidate_lengths), max_input_len),
        dtype=prefix_mask.dtype,
        device=device,
    )
    for slot_index, (candidate_length, explicit_tail_length) in enumerate(
        zip(candidate_lengths, explicit_tail_lengths)
    ):
        row_start = slot_index * batch_size
        row_end = row_start + batch_size
        slot_suffix_start = prefix_len + candidate_length
        slot_suffix_end = slot_suffix_start + suffix_len
        slot_tail_start = slot_suffix_end
        slot_tail_end = slot_tail_start + explicit_tail_length
        input_ids[row_start:row_end, :prefix_len] = prefix_ids
        attention_mask[row_start:row_end, :prefix_len] = prefix_mask
        input_ids[row_start:row_end, prefix_len:slot_suffix_start] = mask_id
        attention_mask[row_start:row_end, prefix_len:slot_suffix_start] = 1
        if suffix_len > 0:
            input_ids[row_start:row_end, slot_suffix_start:slot_suffix_end] = suffix_ids
            attention_mask[row_start:row_end, slot_suffix_start:slot_suffix_end] = (
                suffix_mask
            )
        if explicit_tail_length > 0:
            input_ids[row_start:row_end, slot_tail_start:slot_tail_end] = mask_id
            attention_mask[row_start:row_end, slot_tail_start:slot_tail_end] = 1
    return (input_ids, attention_mask, prefix_len)


def decode_dream_slot_generations_from_sequences(
    tokenizer, sequences, batch_size: int, candidate_lengths, prefix_len: int
):
    slot_generations = []
    for slot_index, candidate_length in enumerate(candidate_lengths):
        row_start = slot_index * batch_size
        row_end = row_start + batch_size
        generated_ids_batch = sequences[
            row_start:row_end, prefix_len : prefix_len + candidate_length
        ]
        token_ids_batch = [
            generated_ids_batch[batch_idx].detach().cpu().tolist()
            for batch_idx in range(batch_size)
        ]
        slot_generations.append(
            {
                "slot_index": slot_index,
                "candidate_length": candidate_length,
                "token_ids_batch": token_ids_batch,
                "completions": [
                    tokenizer.decode(token_ids, skip_special_tokens=True)
                    for token_ids in token_ids_batch
                ],
                "token_strs": [
                    tokenizer.convert_ids_to_tokens(token_ids)
                    for token_ids in token_ids_batch
                ],
            }
        )
    return slot_generations


def _dream_is_low_info_token_id(tokenizer, token_id):
    piece = tokenizer.decode(
        [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return piece.strip() == ""


def _dream_select_suffix_positions_from_token_ids(token_ids, relative_positions):
    if not relative_positions:
        return ([], [])
    valid_positions = []
    selected_ids = []
    token_count = len(token_ids)
    for pos in relative_positions:
        pos = int(pos)
        if 0 <= pos < token_count:
            valid_positions.append(pos)
            selected_ids.append(int(token_ids[pos]))
    return (selected_ids, valid_positions)


def _dream_suffix_positions_to_head_length_or_none(relative_positions):
    if relative_positions is None:
        return None
    normalized = [int(pos) for pos in relative_positions]
    if normalized == list(range(len(normalized))):
        return len(normalized)
    return None


def _dream_sparse_query_positions(prefix_len: int, middle_mask_len: int):
    """
    Dream.generate() left-shifts logits before reading masked positions:
    logits[:, pos] comes from the raw model output at pos-1 (with pos=0 clamped to 0).
    Sparse suffix attention should therefore inspect the source positions that actually
    drive those masked-token predictions, not the mask positions themselves.
    """
    return [
        max(int(abs_pos) - 1, 0)
        for abs_pos in range(int(prefix_len), int(prefix_len) + int(middle_mask_len))
    ]


def _dream_score_source_positions(target_sequence_positions):
    return [max(int(pos) - 1, 0) for pos in target_sequence_positions]


@torch.no_grad()
def collect_dream_sparse_suffix_positions(
    model, tokenizer, device, problems_batch, middle_mask_len, args
):
    sparse_k = max(int(getattr(args, "posthoc_sparse_suffix_k", 0) or 0), 0)
    sparse_window = max(int(getattr(args, "posthoc_sparse_suffix_window", 0) or 0), 0)
    if sparse_k <= 0 or sparse_window <= 0 or int(middle_mask_len) <= 0:
        return [[] for _ in problems_batch]
    prompts = [problem["prompt"] for problem in problems_batch]
    suffixes = [problem["suffix"] for problem in problems_batch]
    encoded_prefix = tokenizer(
        prompts, add_special_tokens=False, padding=True, return_tensors="pt"
    )
    encoded_suffix = tokenizer(
        suffixes, add_special_tokens=False, padding=True, return_tensors="pt"
    )
    prefix_ids = encoded_prefix["input_ids"].to(device)
    prefix_mask = encoded_prefix["attention_mask"].to(device)
    suffix_ids = encoded_suffix["input_ids"].to(device)
    suffix_mask = encoded_suffix["attention_mask"].to(device)
    batch_size, prefix_len = prefix_ids.shape
    suffix_padded_len = suffix_ids.shape[1]
    if suffix_padded_len <= 0:
        return [[] for _ in problems_batch]
    input_ids = torch.full(
        (batch_size, prefix_len + int(middle_mask_len) + suffix_padded_len),
        int(args.mask_id),
        dtype=torch.long,
        device=device,
    )
    input_ids[:, :prefix_len] = prefix_ids
    input_ids[:, prefix_len : prefix_len + int(middle_mask_len)] = int(args.mask_id)
    if suffix_padded_len > 0:
        input_ids[:, prefix_len + int(middle_mask_len) :] = suffix_ids
    attention_mask = torch.cat(
        [
            prefix_mask,
            torch.ones(
                (batch_size, int(middle_mask_len)),
                dtype=prefix_mask.dtype,
                device=device,
            ),
            suffix_mask,
        ],
        dim=1,
    )
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    total_layers = int(
        getattr(model.config, "num_hidden_layers", getattr(model.config, "n_layers", 0))
    )
    target_layer = resolve_attn_layer_index(
        getattr(args, "posthoc_sparse_attn_layer", -1), total_layers
    )
    collector = {
        "enabled": True,
        "target_layer": target_layer,
        "query_positions": _dream_sparse_query_positions(
            prefix_len, int(middle_mask_len)
        ),
        "key_positions": list(
            range(
                prefix_len + int(middle_mask_len),
                prefix_len + int(middle_mask_len) + suffix_padded_len,
            )
        ),
    }
    modules = install_light_attn_collector(model, collector)
    try:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
    finally:
        clear_light_attn_collector(modules)
    attention_scores = collector.get("scores")
    if attention_scores is None:
        return [[] for _ in problems_batch]
    base_suffix_positions = resolve_base_suffix_positions(args)
    score_rows = attention_scores.detach().cpu()
    suffix_token_id_lists = [
        tokenizer(problem["suffix"], add_special_tokens=False)["input_ids"]
        for problem in problems_batch
    ]
    selected_positions_by_sample = []
    for batch_idx, token_ids in enumerate(suffix_token_id_lists):
        valid_len = int(suffix_mask[batch_idx].to(torch.long).sum().item())
        candidate_limit = min(len(token_ids), valid_len, sparse_window)
        if candidate_limit <= 0:
            selected_positions_by_sample.append([])
            continue
        start_offset = max(suffix_padded_len - valid_len, 0)
        excluded_positions = set()
        if getattr(args, "posthoc_sparse_suffix_disjoint_from_base", False):
            excluded_positions = {int(pos) for pos in base_suffix_positions}
        ranked = []
        for rel_pos in range(candidate_limit):
            if rel_pos in excluded_positions:
                continue
            token_id = int(token_ids[rel_pos])
            if getattr(
                args, "posthoc_sparse_exclude_low_info", False
            ) and _dream_is_low_info_token_id(tokenizer, token_id):
                continue
            abs_pos = start_offset + rel_pos
            if abs_pos < 0 or abs_pos >= score_rows.shape[1]:
                continue
            ranked.append((float(score_rows[batch_idx, abs_pos].item()), rel_pos))
        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen = sorted((rel_pos for _, rel_pos in ranked[:sparse_k]))
        selected_positions_by_sample.append(chosen)
    return selected_positions_by_sample


@torch.no_grad()
def evaluate_dream_candidate_confidence(
    model,
    tokenizer,
    device,
    problems_batch,
    slot_generations,
    mask_id,
    score_basis="target_logprob",
    score_topk=32,
    suffix_relative_positions_by_sample=None,
    full_suffix_context=False,
):
    """
    Score each Dream candidate slot with MASK tokens at the slot positions.

    We intentionally keep Dream's bidirectional visibility for the suffix side:
    when scoring token t_i, the source hidden state may still see later visible
    suffix tokens (including the immediate next token). That is expected for
    Dream and should not be "fixed" into LLaDA-style causal behavior. The only
    thing we prevent is putting the candidate slot tokens themselves into the
    scored slot positions, because that would let the model read the answer it
    is supposed to evaluate.
    """
    batch_size = len(problems_batch)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if suffix_relative_positions_by_sample is None:
        suffix_relative_positions_by_sample = [[] for _ in problems_batch]
    prefix_id_lists = [
        tokenizer(problem["prompt"], add_special_tokens=False)["input_ids"]
        for problem in problems_batch
    ]
    suffix_token_id_lists = [
        tokenizer(problem["suffix"], add_special_tokens=False)["input_ids"]
        for problem in problems_batch
    ]
    suffix_selected_positions_lists = []
    for suffix_token_ids, requested_positions in zip(
        suffix_token_id_lists, suffix_relative_positions_by_sample
    ):
        _selected_ids, selected_positions = (
            _dream_select_suffix_positions_from_token_ids(
                suffix_token_ids, requested_positions
            )
        )
        suffix_selected_positions_lists.append(selected_positions)
    flat_entries = []
    full_sequences = []
    full_position_ids = []
    total_lengths = []
    per_slot_outputs = {
        int(slot_generation["slot_index"]): [
            {"token_conf": [], "token_score": []} for _ in range(batch_size)
        ]
        for slot_generation in slot_generations
    }
    for slot_generation in slot_generations:
        slot_index = int(slot_generation["slot_index"])
        generated_token_ids_batch = slot_generation["token_ids_batch"]
        for batch_idx, token_ids in enumerate(generated_token_ids_batch):
            prefix_ids = list(prefix_id_lists[batch_idx])
            suffix_token_ids = list(suffix_token_id_lists[batch_idx])
            suffix_selected_positions = list(suffix_selected_positions_lists[batch_idx])
            slot_token_ids = [int(token_id) for token_id in token_ids]
            prefix_len = len(prefix_ids)
            slot_len = len(slot_token_ids)
            mask_slot = [int(mask_id)] * slot_len
            if full_suffix_context:
                full_sequence = prefix_ids + mask_slot + suffix_token_ids
                sequence_position_ids = list(range(len(full_sequence)))
                target_ids = list(slot_token_ids)
                target_sequence_positions = list(
                    range(prefix_len, prefix_len + slot_len)
                )
                for pos in suffix_selected_positions:
                    pos = int(pos)
                    if 0 <= pos < len(suffix_token_ids):
                        target_ids.append(int(suffix_token_ids[pos]))
                        target_sequence_positions.append(prefix_len + slot_len + pos)
                score_source_positions = _dream_score_source_positions(
                    target_sequence_positions
                )
            else:
                suffix_selected_ids = [
                    int(suffix_token_ids[pos])
                    for pos in suffix_selected_positions
                    if 0 <= int(pos) < len(suffix_token_ids)
                ]
                target_ids = list(slot_token_ids) + suffix_selected_ids
                full_sequence = prefix_ids + mask_slot + suffix_selected_ids
                contiguous_head_length = _dream_suffix_positions_to_head_length_or_none(
                    suffix_selected_positions
                )
                if contiguous_head_length is not None:
                    suffix_position_ids = list(
                        range(
                            prefix_len + slot_len,
                            prefix_len + slot_len + len(suffix_selected_positions),
                        )
                    )
                else:
                    suffix_position_ids = [
                        prefix_len + slot_len + int(pos)
                        for pos in suffix_selected_positions
                    ]
                sequence_position_ids = (
                    list(range(prefix_len + slot_len)) + suffix_position_ids
                )
                target_sequence_positions = list(
                    range(prefix_len, prefix_len + slot_len + len(suffix_selected_ids))
                )
                score_source_positions = _dream_score_source_positions(
                    target_sequence_positions
                )
            flat_entries.append(
                {
                    "slot_index": slot_index,
                    "batch_idx": batch_idx,
                    "target_ids": target_ids,
                    "score_source_positions": score_source_positions,
                    "total_len": len(full_sequence),
                }
            )
            full_sequences.append(full_sequence)
            full_position_ids.append(sequence_position_ids)
            total_lengths.append(len(full_sequence))
    if not flat_entries:
        return []
    max_seq_len = max(total_lengths)
    input_ids = torch.full(
        (len(full_sequences), max_seq_len),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(full_sequences), max_seq_len), dtype=torch.long, device=device
    )
    position_ids = torch.zeros(
        (len(full_sequences), max_seq_len), dtype=torch.long, device=device
    )
    for flat_idx, sequence in enumerate(full_sequences):
        start = max_seq_len - len(sequence)
        input_ids[flat_idx, start:] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        attention_mask[flat_idx, start:] = 1
        position_ids[flat_idx, start:] = torch.tensor(
            full_position_ids[flat_idx], dtype=torch.long, device=device
        )
    logits = model(
        input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
    ).logits.float()
    probs = F.softmax(logits, dim=-1)
    logprobs = F.log_softmax(logits, dim=-1)
    for flat_idx, entry in enumerate(flat_entries):
        slot_index = int(entry["slot_index"])
        batch_idx = int(entry["batch_idx"])
        target_ids = entry["target_ids"]
        if not target_ids:
            continue
        pad_len = max_seq_len - int(entry["total_len"])
        source_positions = torch.tensor(
            [pad_len + int(pos) for pos in entry["score_source_positions"]],
            dtype=torch.long,
            device=device,
        )
        relevant_probs = probs[flat_idx].index_select(0, source_positions)
        relevant_logprobs = logprobs[flat_idx].index_select(0, source_positions)
        target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device)
        token_conf = torch.gather(
            relevant_probs, dim=-1, index=target_tensor.unsqueeze(-1)
        ).squeeze(-1)
        token_logprob = torch.gather(
            relevant_logprobs, dim=-1, index=target_tensor.unsqueeze(-1)
        ).squeeze(-1)
        if score_basis == "target_logprob":
            token_score = token_logprob
        elif score_basis == "predictive_entropy":
            token_score = (relevant_probs * relevant_logprobs).sum(dim=-1)
        elif score_basis == "topk_mass":
            k = min(int(score_topk), relevant_probs.shape[-1])
            if k <= 0:
                raise ValueError(
                    "posthoc_score_topk must be positive when posthoc_score_basis='topk_mass'."
                )
            token_score = torch.topk(relevant_probs, k=k, dim=-1).values.sum(dim=-1)
        elif score_basis == "topk_logprob_sum":
            k = min(int(score_topk), relevant_logprobs.shape[-1])
            if k <= 0:
                raise ValueError(
                    "posthoc_score_topk must be positive when posthoc_score_basis='topk_logprob_sum'."
                )
            token_score = torch.topk(relevant_logprobs, k=k, dim=-1).values.sum(dim=-1)
        elif score_basis == "topk_entropy":
            k = min(int(score_topk), relevant_probs.shape[-1])
            if k <= 0:
                raise ValueError(
                    "posthoc_score_topk must be positive when posthoc_score_basis='topk_entropy'."
                )
            topk_probs = torch.topk(relevant_probs, k=k, dim=-1).values
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(
                torch.finfo(topk_probs.dtype).tiny
            )
            token_score = (
                topk_probs
                * topk_probs.clamp_min(torch.finfo(topk_probs.dtype).tiny).log()
            ).sum(dim=-1)
        else:
            raise ValueError(f"Unsupported score_basis: {score_basis}")
        per_slot_outputs[slot_index][batch_idx]["token_conf"] = (
            token_conf.detach().cpu().tolist()
        )
        per_slot_outputs[slot_index][batch_idx]["token_score"] = (
            token_score.detach().cpu().tolist()
        )
    candidate_confidence_records = []
    for slot_generation in slot_generations:
        slot_index = int(slot_generation["slot_index"])
        batch_outputs = per_slot_outputs[slot_index]
        max_eval_len = max(
            (len(row["token_score"]) for row in batch_outputs), default=0
        )
        token_conf = torch.zeros(
            (batch_size, max_eval_len), dtype=torch.float32, device=device
        )
        token_score = torch.zeros(
            (batch_size, max_eval_len), dtype=torch.float32, device=device
        )
        valid_mask = torch.zeros(
            (batch_size, max_eval_len), dtype=torch.bool, device=device
        )
        avg_conf = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        min_conf = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        avg_score = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        min_score = torch.zeros((batch_size,), dtype=torch.float32, device=device)
        for batch_idx, row in enumerate(batch_outputs):
            score_list = [float(value) for value in row["token_score"]]
            conf_list = [float(value) for value in row["token_conf"]]
            row_len = len(score_list)
            if row_len <= 0:
                continue
            token_conf[batch_idx, :row_len] = torch.tensor(
                conf_list, dtype=torch.float32, device=device
            )
            token_score[batch_idx, :row_len] = torch.tensor(
                score_list, dtype=torch.float32, device=device
            )
            valid_mask[batch_idx, :row_len] = True
            avg_conf[batch_idx] = float(sum(conf_list) / row_len)
            min_conf[batch_idx] = float(min(conf_list))
            avg_score[batch_idx] = float(sum(score_list) / row_len)
            min_score[batch_idx] = float(min(score_list))
        candidate_confidence_records.append(
            {
                "slot_name": f"front_{slot_index}",
                "token_conf": token_conf,
                "token_score": token_score,
                "valid_mask": valid_mask,
                "avg_conf": avg_conf,
                "min_conf": min_conf,
                "avg_score": avg_score,
                "min_score": min_score,
                "score_basis": score_basis,
            }
        )
    return candidate_confidence_records


def run_generation_batch_dream(
    model, tokenizer, device, problems_batch, middle_mask_len, front_mask_lengths, args
):
    candidate_lengths = [int(length) for length in front_mask_lengths]
    dream_slot_decode_mode = "parallel_slots"
    suppress_token_ids = (
        get_special_token_suppress_ids(tokenizer, mask_id=args.mask_id)
        if getattr(args, "suppress_special_tokens", False)
        else []
    )

    def suppress_dream_logits_hook(step, x, logits):
        if suppress_token_ids:
            if logits.dim() == 3:
                logits[:, :, suppress_token_ids] = -torch.inf
            else:
                logits[:, suppress_token_ids] = -torch.inf
        return logits

    slot_generations = []
    generation_start = time.perf_counter()
    input_ids, attention_mask, prefix_len = build_dream_parallel_generation_batch(
        tokenizer=tokenizer,
        problems_batch=problems_batch,
        candidate_lengths=candidate_lengths,
        mask_id=args.mask_id,
        device=device,
    )
    gen_config = DreamGenerationConfig(
        max_length=input_ids.shape[1] + 1,
        max_new_tokens=1,
        steps=args.steps
        if args.steps is not None
        else max(candidate_lengths + [int(middle_mask_len)]),
        temperature=args.temperature,
        alg=args.dream_alg,
        top_p=args.dream_top_p,
        top_k=args.dream_top_k,
        mask_token_id=args.mask_id,
        pad_token_id=tokenizer.pad_token_id,
        return_dict_in_generate=True,
        output_history=False,
    )
    output = model.generate(
        inputs=input_ids,
        attention_mask=attention_mask,
        generation_config=gen_config,
        generation_tokens_hook_func=lambda step, x, logits: x,
        generation_logits_hook_func=suppress_dream_logits_hook,
    )
    sequences = output.sequences if hasattr(output, "sequences") else output
    slot_generations.extend(
        decode_dream_slot_generations_from_sequences(
            tokenizer=tokenizer,
            sequences=sequences,
            batch_size=len(problems_batch),
            candidate_lengths=candidate_lengths,
            prefix_len=prefix_len,
        )
    )
    generation_time = time.perf_counter() - generation_start
    selection_time = 0.0
    selection_start = time.perf_counter()
    base_suffix_positions = resolve_base_suffix_positions(args)
    sparse_suffix_positions_by_sample = collect_dream_sparse_suffix_positions(
        model=model,
        tokenizer=tokenizer,
        device=device,
        problems_batch=problems_batch,
        middle_mask_len=middle_mask_len,
        args=args,
    )
    effective_suffix_positions_by_sample = [
        merge_suffix_positions(base_suffix_positions, sparse_positions)
        for sparse_positions in sparse_suffix_positions_by_sample
    ]
    suffix_selection_details_by_sample = [
        build_suffix_selection_details(
            problem=problem,
            tokenizer=tokenizer,
            final_positions=effective_suffix_positions_by_sample[batch_idx],
            sparse_positions=sparse_suffix_positions_by_sample[batch_idx],
        )
        for batch_idx, problem in enumerate(problems_batch)
    ]
    selection_time += time.perf_counter() - selection_start
    skip_posthoc_candidate_scoring = False
    selection_start = time.perf_counter()
    candidate_confidence_records = evaluate_dream_candidate_confidence(
        model=model,
        tokenizer=tokenizer,
        device=device,
        problems_batch=problems_batch,
        slot_generations=slot_generations,
        mask_id=args.mask_id,
        score_basis=getattr(args, "posthoc_score_basis", "target_logprob"),
        score_topk=getattr(args, "posthoc_score_topk", 32),
        suffix_relative_positions_by_sample=effective_suffix_positions_by_sample,
        full_suffix_context=bool(getattr(args, "posthoc_full_suffix_context", False)),
    )
    selection_time += time.perf_counter() - selection_start
    generated_samples = []
    for batch_idx, problem in enumerate(problems_batch):
        all_front_completions = [
            sg["completions"][batch_idx] for sg in slot_generations
        ]
        all_front_slot_token_ids = [
            sg["token_ids_batch"][batch_idx] for sg in slot_generations
        ]
        all_front_slot_token_strs = [
            sg["token_strs"][batch_idx] for sg in slot_generations
        ]
        selected_slot_indices = list(range(len(candidate_lengths)))
        if False and True and selected_slot_indices:
            if getattr(args, "posthoc_filter_invalid_ast", False):
                for slot_index in selected_slot_indices:
                    slot_completion = slot_generations[slot_index]["completions"][
                        batch_idx
                    ]
                    if candidate_has_valid_python_ast(problem, slot_completion):
                        selected_slot_indices = [slot_index]
                        break
                else:
                    selected_slot_indices = [selected_slot_indices[0]]
            else:
                selected_slot_indices = [selected_slot_indices[0]]
        elif True and candidate_confidence_records:
            selection_start = time.perf_counter()
            front_posthoc_score_by_slot, _, _ = build_posthoc_score_by_slot(
                candidate_confidence_records, batch_idx, candidate_lengths
            )
            score_list = []
            ast_valid_slot_indices = []
            eligible_slot_indices = []
            special_clean_slot_indices = []
            ast_filter_enabled = bool(
                getattr(args, "posthoc_filter_invalid_ast", False)
            )
            for slot_index, slot_length in enumerate(candidate_lengths):
                token_scores = front_posthoc_score_by_slot[slot_index]
                slot_token_ids = slot_generations[slot_index]["token_ids_batch"][
                    batch_idx
                ]
                if ast_filter_enabled:
                    slot_completion = slot_generations[slot_index]["completions"][
                        batch_idx
                    ]
                    if not candidate_has_valid_python_ast(problem, slot_completion):
                        score_list.append(float("-inf"))
                        continue
                    ast_valid_slot_indices.append(slot_index)
                has_suppressed_token = candidate_has_suppressed_token(
                    slot_token_ids, suppress_token_ids
                )
                score_list.append(
                    compute_slot_selection_score(
                        token_scores=token_scores,
                        slot_length=slot_length,
                        metric=args.posthoc_select_metric,
                        alpha=args.posthoc_select_alpha,
                        gamma=args.posthoc_select_gamma,
                        slot_metric=args.posthoc_select_slot_metric,
                        suffix_metric=args.posthoc_select_suffix_metric,
                        addition_metric=args.posthoc_select_addition_metric,
                        beta=args.posthoc_select_beta,
                    )
                )
                if not has_suppressed_token:
                    special_clean_slot_indices.append(slot_index)
                if passes_token_score_threshold(
                    token_scores, args.posthoc_token_score_threshold
                ) and (not has_suppressed_token):
                    eligible_slot_indices.append(slot_index)
            if score_list:
                ast_candidate_indices = (
                    ast_valid_slot_indices
                    if ast_filter_enabled and ast_valid_slot_indices
                    else list(range(len(score_list)))
                )
                eligible_slot_indices = [
                    idx for idx in eligible_slot_indices if idx in ast_candidate_indices
                ]
                special_clean_slot_indices = [
                    idx
                    for idx in special_clean_slot_indices
                    if idx in ast_candidate_indices
                ]
                candidate_pool = (
                    eligible_slot_indices
                    if eligible_slot_indices
                    else special_clean_slot_indices or ast_candidate_indices
                )
                ranked_slot_indices = sorted(
                    candidate_pool, key=lambda i: score_list[i], reverse=True
                )
                selected_slot_indices = [int(ranked_slot_indices[0])]
            selection_time += time.perf_counter() - selection_start
        for slot_index in selected_slot_indices:
            slot_generation = slot_generations[slot_index]
            token_ids = slot_generation["token_ids_batch"][batch_idx]
            generated_samples.append(
                {
                    "task_id": problem["task_id"],
                    "completion": slot_generation["completions"][batch_idx],
                    "mask_token_length": int(slot_generation["candidate_length"]),
                    "slot_index": int(slot_index),
                    "front_mask_lengths": list(candidate_lengths),
                    "middle_mask_len": int(middle_mask_len),
                    "generated_token_count": len(token_ids),
                    "generated_token_ids": token_ids,
                    "generated_tokens": slot_generation["token_strs"][batch_idx],
                    "model_backend": "dream",
                    "dream_slot_decode_mode": "parallel_slots",
                    "all_front_completions": all_front_completions,
                    "all_front_slot_token_ids": all_front_slot_token_ids,
                    "all_front_slot_tokens": all_front_slot_token_strs,
                    "posthoc_full_suffix_context": bool(
                        getattr(args, "posthoc_full_suffix_context", False)
                    ),
                    "posthoc_suffix_relative_positions": list(
                        suffix_selection_details_by_sample[batch_idx]["positions"]
                    ),
                    "posthoc_sparse_suffix_positions": list(
                        suffix_selection_details_by_sample[batch_idx][
                            "sparse_positions"
                        ]
                    ),
                    "posthoc_suffix_selected_token_ids": list(
                        suffix_selection_details_by_sample[batch_idx]["token_ids"]
                    ),
                    "posthoc_suffix_selected_tokens": list(
                        suffix_selection_details_by_sample[batch_idx]["tokens"]
                    ),
                    "posthoc_suffix_selected_pieces": list(
                        suffix_selection_details_by_sample[batch_idx]["pieces"]
                    ),
                }
            )
            if getattr(args, "show_posthoc_suffix_selection", False):
                print_suffix_selection_details(
                    problem["task_id"], suffix_selection_details_by_sample[batch_idx]
                )
    add_stage_timing(
        args, generation_seconds=generation_time, selection_seconds=selection_time
    )
    return generated_samples
