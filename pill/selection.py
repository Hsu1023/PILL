"""PILL inference routines extracted from the original implementation."""

import os, ast, contextlib, subprocess, tempfile, time
import torch
import numpy as np
from .syntax import JAVA_PAIR_STUB, problem_language, build_external_program
from .decoding import (
    collect_generate_mask_first_pass_suffix_attention_scores,
    evaluate_front_candidate_causal_confidence,
    generate_mask,
)


def get_special_token_suppress_ids(tokenizer, mask_id=None):
    token_ids = set()
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        if token_id is None:
            continue
        token_ids.add(int(token_id))
    if mask_id is not None:
        token_ids.add(int(mask_id))
    return sorted(token_ids)


def candidate_has_suppressed_token(token_ids, suppress_token_ids):
    if not suppress_token_ids:
        return False
    suppressed = {int(token_id) for token_id in suppress_token_ids}
    return any((int(token_id) in suppressed for token_id in token_ids))


def _candidate_compiles_as_cpp(program):
    with tempfile.TemporaryDirectory() as dirname:
        source_path = os.path.join(dirname, "main.cpp")
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write(program)
        result = subprocess.run(
            ["g++", "-std=c++17", "-fsyntax-only", source_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            text=True,
        )
        return result.returncode == 0


def _candidate_compiles_as_java(program):
    with tempfile.TemporaryDirectory() as dirname:
        pair_dir = os.path.join(dirname, "org", "javatuples")
        os.makedirs(pair_dir, exist_ok=True)
        pair_path = os.path.join(pair_dir, "Pair.java")
        source_path = os.path.join(dirname, "Problem.java")
        with open(pair_path, "w", encoding="utf-8") as handle:
            handle.write(JAVA_PAIR_STUB + "\n")
        with open(source_path, "w", encoding="utf-8") as handle:
            handle.write(program)
        result = subprocess.run(
            ["javac", source_path, pair_path],
            cwd=dirname,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
            text=True,
        )
        return result.returncode == 0


def candidate_has_valid_python_ast(problem, completion):
    if problem is None:
        return False
    language = problem_language(problem)
    try:
        if language in {"python", "py"}:
            ast.parse(
                str(problem.get("prompt", ""))
                + str(completion)
                + str(problem.get("suffix", ""))
            )
            return True
        program = build_external_program(problem, str(completion))
        if language in {"cpp", "c++", "c"}:
            return _candidate_compiles_as_cpp(program)
        if language == "java":
            return _candidate_compiles_as_java(program)
        return True
    except (SyntaxError, subprocess.SubprocessError, OSError):
        return False
    except (TypeError, ValueError):
        return False


def resolve_base_suffix_positions(args):
    if getattr(args, "posthoc_suffix_pos_id", None) is not None:
        return [int(pos) for pos in args.posthoc_suffix_pos_id]
    return list(range(max(int(getattr(args, "posthoc_suffix_head_length", 0) or 0), 0)))


def merge_suffix_positions(base_positions, extra_positions):
    merged = sorted(
        {
            int(pos)
            for pos in (base_positions or []) + (extra_positions or [])
            if int(pos) >= 0
        }
    )
    return merged


def resolve_attn_layer_index(requested_layer: int, total_layers: int) -> int:
    total_layers = int(total_layers)
    if total_layers <= 0:
        return 0
    layer_idx = int(requested_layer)
    if layer_idx < 0:
        layer_idx = total_layers + layer_idx
    return max(0, min(layer_idx, total_layers - 1))


def _iter_light_attn_modules(model):
    for module in model.modules():
        if module.__class__.__name__ in {
            "LLaDASequentialBlock",
            "LLaDALlamaBlock",
            "LLaDAMoEAttention",
            "LLaDAMoESdpaAttention",
            "LLaDAMoEFlashAttention2",
            "DreamAttention",
            "DreamSdpaAttention",
        }:
            yield module


def install_light_attn_collector(model, collector):
    modules = list(_iter_light_attn_modules(model))
    for module in modules:
        setattr(module, "_light_attn_collector", collector)
    return modules


def clear_light_attn_collector(modules):
    for module in modules:
        if hasattr(module, "_light_attn_collector"):
            delattr(module, "_light_attn_collector")
        for handle in getattr(module, "_light_attn_collector_handles", []):
            handle.remove()
        if hasattr(module, "_light_attn_collector_handles"):
            delattr(module, "_light_attn_collector_handles")


def decode_token_piece(tokenizer, token_id):
    return tokenizer.decode(
        [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def is_low_info_token_id(tokenizer, token_id):
    piece = decode_token_piece(tokenizer, token_id)
    return piece.strip() == ""


def build_suffix_selection_details(
    problem, tokenizer, final_positions, sparse_positions
):
    suffix_token_ids = tokenizer(problem["suffix"], add_special_tokens=False)[
        "input_ids"
    ]
    final_positions = [int(pos) for pos in final_positions]
    sparse_positions = [int(pos) for pos in sparse_positions]
    selected_token_ids = []
    selected_tokens = []
    selected_pieces = []
    for pos in final_positions:
        if 0 <= pos < len(suffix_token_ids):
            token_id = int(suffix_token_ids[pos])
            selected_token_ids.append(token_id)
            selected_tokens.append(tokenizer.convert_ids_to_tokens([token_id])[0])
            selected_pieces.append(decode_token_piece(tokenizer, token_id))
    return {
        "positions": final_positions,
        "sparse_positions": sparse_positions,
        "token_ids": selected_token_ids,
        "tokens": selected_tokens,
        "pieces": selected_pieces,
    }


def print_suffix_selection_details(task_id, details):
    token_items = " ".join(
        (
            f"{pos}:{repr(piece)}"
            for pos, piece in zip(details["positions"], details["pieces"])
        )
    )
    print(
        f"[posthoc_suffix_selection] task={task_id} final={details['positions']} sparse={details['sparse_positions']} tokens={token_items}"
    )


def add_stage_timing(
    args, generation_seconds: float = 0.0, selection_seconds: float = 0.0
):
    timing = getattr(args, "_stage_timing", None)
    if timing is None:
        return
    timing["generation_seconds"] = float(timing.get("generation_seconds", 0.0)) + float(
        generation_seconds
    )
    timing["selection_seconds"] = float(timing.get("selection_seconds", 0.0)) + float(
        selection_seconds
    )


def select_sparse_suffix_positions_from_attention(
    tokenizer,
    suffix_token_id_lists,
    attention_scores,
    window: int,
    k: int,
    exclude_low_info: bool,
    excluded_positions_by_sample=None,
):
    if attention_scores is None or k <= 0 or window <= 0:
        return [[] for _ in suffix_token_id_lists]
    selected_positions_by_sample = []
    score_rows = attention_scores.detach().cpu()
    for batch_idx, token_ids in enumerate(suffix_token_id_lists):
        if batch_idx >= score_rows.shape[0]:
            selected_positions_by_sample.append([])
            continue
        candidate_limit = min(len(token_ids), score_rows.shape[1], max(int(window), 0))
        excluded_positions = set()
        if excluded_positions_by_sample is not None and batch_idx < len(
            excluded_positions_by_sample
        ):
            excluded_positions = {
                int(pos) for pos in excluded_positions_by_sample[batch_idx]
            }
        ranked = []
        for rel_pos in range(candidate_limit):
            if rel_pos in excluded_positions:
                continue
            token_id = int(token_ids[rel_pos])
            if exclude_low_info and is_low_info_token_id(tokenizer, token_id):
                continue
            ranked.append((float(score_rows[batch_idx, rel_pos].item()), rel_pos))
        ranked.sort(key=lambda item: item[0], reverse=True)
        chosen = sorted((rel_pos for _, rel_pos in ranked[: max(int(k), 0)]))
        selected_positions_by_sample.append(chosen)
    return selected_positions_by_sample


def extract_suffix_token_id_lists(suffix_ids, suffix_mask, window: int):
    if suffix_ids is None:
        return []
    token_id_lists = []
    suffix_ids_cpu = suffix_ids.detach().cpu()
    suffix_mask_cpu = suffix_mask.detach().cpu() if suffix_mask is not None else None
    for row_idx in range(suffix_ids_cpu.shape[0]):
        token_row = suffix_ids_cpu[row_idx].tolist()
        if suffix_mask_cpu is None:
            valid_token_ids = token_row
        else:
            mask_row = suffix_mask_cpu[row_idx].tolist()
            valid_token_ids = [
                int(token_id) for token_id, keep in zip(token_row, mask_row) if keep
            ]
        token_id_lists.append(valid_token_ids[: max(int(window), 0)])
    return token_id_lists


def collect_llada_sparse_suffix_positions(
    model,
    tokenizer,
    problems_batch,
    prefix_ids,
    prefix_mask,
    suffix_ids,
    suffix_mask,
    front_mask_lengths,
    middle_mask_len,
    args,
):
    sparse_k = max(int(getattr(args, "posthoc_sparse_suffix_k", 0) or 0), 0)
    sparse_window = max(int(getattr(args, "posthoc_sparse_suffix_window", 0) or 0), 0)
    if sparse_k <= 0 or sparse_window <= 0:
        return [[] for _ in problems_batch]
    base_suffix_positions = resolve_base_suffix_positions(args)
    total_suffix_len = suffix_ids.shape[1] if suffix_ids is not None else 0
    if total_suffix_len <= 0 or int(middle_mask_len) <= 0:
        return [[] for _ in problems_batch]
    suffix_window = min(sparse_window, total_suffix_len)
    total_layers = int(
        getattr(model.config, "n_layers", getattr(model.config, "num_hidden_layers", 0))
    )
    target_layer = resolve_attn_layer_index(
        getattr(args, "posthoc_sparse_attn_layer", -1), total_layers
    )
    collector = {
        "enabled": True,
        "target_layer": target_layer,
        "query_positions": list(
            range(
                sum((int(v) for v in front_mask_lengths)) + prefix_ids.shape[1],
                sum((int(v) for v in front_mask_lengths))
                + prefix_ids.shape[1]
                + int(middle_mask_len),
            )
        ),
        "key_positions": list(
            range(
                sum((int(v) for v in front_mask_lengths))
                + prefix_ids.shape[1]
                + int(middle_mask_len),
                sum((int(v) for v in front_mask_lengths))
                + prefix_ids.shape[1]
                + int(middle_mask_len)
                + suffix_window,
            )
        ),
    }
    modules = install_light_attn_collector(model, collector)
    try:
        collect_generate_mask_first_pass_suffix_attention_scores(
            model=model,
            prefix_ids=prefix_ids,
            front_mask_lengths=front_mask_lengths,
            suffix_ids=suffix_ids,
            attention_mask=prefix_mask,
            suffix_attention_mask=suffix_mask,
            gen_length=int(middle_mask_len),
            cfg_scale=args.cfg_scale,
            mask_id=args.mask_id,
            position_mode=args.position_mode,
        )
    finally:
        clear_light_attn_collector(modules)
    attention_scores = collector.get("scores")
    if attention_scores is None:
        return [[] for _ in problems_batch]
    expected_batch = len(problems_batch)
    if (
        attention_scores.shape[0] >= expected_batch * 2
        and getattr(args, "cfg_scale", 0.0) > 0.0
    ):
        attention_scores = attention_scores[:expected_batch]
    suffix_token_id_lists = extract_suffix_token_id_lists(
        suffix_ids=suffix_ids, suffix_mask=suffix_mask, window=suffix_window
    )
    excluded_positions_by_sample = None
    if getattr(args, "posthoc_sparse_suffix_disjoint_from_base", False):
        excluded_positions_by_sample = [
            list(base_suffix_positions) for _ in problems_batch
        ]
    return select_sparse_suffix_positions_from_attention(
        tokenizer=tokenizer,
        suffix_token_id_lists=suffix_token_id_lists,
        attention_scores=attention_scores,
        window=suffix_window,
        k=sparse_k,
        exclude_low_info=bool(getattr(args, "posthoc_sparse_exclude_low_info", False)),
        excluded_positions_by_sample=excluded_positions_by_sample,
    )


def resolve_front_mask_lengths(args):
    if args.front_mask_lengths:
        return [int(length) for length in args.front_mask_lengths]
    return list(range(args.min_mask_len, args.max_mask_len + 1))


def get_mask_regions(out, prefix_ids, front_mask_lengths, middle_mask_len):
    total_front_len = sum(front_mask_lengths)
    prefix_len = prefix_ids.shape[1]
    middle_start = total_front_len + prefix_len
    middle_end = middle_start + middle_mask_len
    front_ids = out[:, :total_front_len]
    middle_ids = out[:, middle_start:middle_end]
    return (front_ids, middle_ids)


def mean_or_none(values):
    if not values:
        return None
    return float(sum(values) / len(values))


def min_or_none(values):
    if not values:
        return None
    return float(min(values))


def weighted_combine(slot_value, suffix_value, alpha: float):
    if slot_value is None and suffix_value is None:
        return float("-inf")
    if slot_value is None:
        return float(suffix_value)
    if suffix_value is None:
        return float(slot_value)
    return float(alpha * slot_value + (1.0 - alpha) * suffix_value)


def passes_token_score_threshold(token_scores, threshold):
    if threshold is None:
        return True
    return all((float(score) >= float(threshold) for score in token_scores))


def compute_slot_selection_score(
    token_scores, slot_length, metric="slot_suffix_avg_avg", alpha=0.5, **kwargs
):
    """Combine mean completion and suffix log probabilities."""
    if metric != "slot_suffix_avg_avg":
        raise ValueError("Only the PILL selection score is supported.")
    slot = mean_or_none([float(v) for v in token_scores[:slot_length]])
    suffix = mean_or_none([float(v) for v in token_scores[slot_length:]])
    return weighted_combine(slot, suffix, alpha)


def build_posthoc_score_by_slot(
    candidate_confidence_records, batch_idx, front_mask_lengths
):
    posthoc_score_by_slot = {}
    posthoc_avg_score_by_slot = {}
    posthoc_min_score_by_slot = {}
    for record in candidate_confidence_records:
        slot_index = int(record["slot_name"].split("_", maxsplit=1)[1])
        valid_mask = record.get("valid_mask")
        if valid_mask is not None:
            valid_mask_row = valid_mask[batch_idx].detach().cpu().tolist()
            token_score_row = record["token_score"][batch_idx].detach().cpu().tolist()
            filtered_score = [
                float(v) for v, keep in zip(token_score_row, valid_mask_row) if keep
            ]
        else:
            filtered_score = record["token_score"][batch_idx].detach().cpu().tolist()
        posthoc_score_by_slot[slot_index] = filtered_score
        posthoc_avg_score_by_slot[slot_index] = float(
            record["avg_score"][batch_idx].item()
        )
        posthoc_min_score_by_slot[slot_index] = float(
            record["min_score"][batch_idx].item()
        )
    for slot_index, slot_length in enumerate(front_mask_lengths):
        if slot_index not in posthoc_score_by_slot:
            posthoc_score_by_slot[slot_index] = [0.0] * slot_length
            posthoc_avg_score_by_slot[slot_index] = 0.0
            posthoc_min_score_by_slot[slot_index] = 0.0
    return (posthoc_score_by_slot, posthoc_avg_score_by_slot, posthoc_min_score_by_slot)


def build_sample(problem, slot_index, slot_length, completion):
    return {
        "task_id": problem["task_id"],
        "completion": completion,
        "mask_token_length": slot_length,
        "slot_index": slot_index,
    }


def run_generation_batch(model, tokenizer, device, problems_batch, mask_len, args):
    prompts = [problem["prompt"] for problem in problems_batch]
    suffixes = [problem["suffix"] for problem in problems_batch]
    suppress_token_ids = (
        get_special_token_suppress_ids(tokenizer, mask_id=args.mask_id)
        if getattr(args, "suppress_special_tokens", False)
        else []
    )
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
    front_mask_lengths = resolve_front_mask_lengths(args)
    if not front_mask_lengths:
        raise ValueError("front_mask_lengths must be non-empty")
    middle_mask_len = (
        args.middle_mask_len
        if args.middle_mask_len is not None
        else max(front_mask_lengths)
    )
    decode_steps = (
        args.steps
        if args.steps is not None
        else max(front_mask_lengths + [middle_mask_len])
    )
    skip_posthoc_candidate_scoring = False
    selection_time = 0.0
    selection_start = time.perf_counter()
    base_suffix_positions = resolve_base_suffix_positions(args)
    sparse_suffix_positions_by_sample = collect_llada_sparse_suffix_positions(
        model=model,
        tokenizer=tokenizer,
        problems_batch=problems_batch,
        prefix_ids=prefix_ids,
        prefix_mask=prefix_mask,
        suffix_ids=suffix_ids,
        suffix_mask=suffix_mask,
        front_mask_lengths=front_mask_lengths,
        middle_mask_len=middle_mask_len,
        args=args,
    )
    effective_suffix_positions_by_sample = [
        merge_suffix_positions(base_suffix_positions, sparse_positions)
        for sparse_positions in sparse_suffix_positions_by_sample
    ]
    selection_time += time.perf_counter() - selection_start
    with (
        open(os.devnull, "w", encoding="utf-8") as sink,
        contextlib.redirect_stdout(sink),
    ):
        generation_start = time.perf_counter()
        out = generate_mask(
            model,
            prefix_ids=prefix_ids,
            front_mask_lengths=front_mask_lengths,
            suffix_ids=suffix_ids,
            attention_mask=prefix_mask,
            suffix_attention_mask=suffix_mask,
            steps=decode_steps,
            gen_length=middle_mask_len,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            mask_id=args.mask_id,
            logits_eos_inf=args.logits_eos_inf,
            confidence_eos_eot_inf=args.confidence_eos_eot_inf,
            position_mode=args.position_mode,
            return_confidence=False,
            tokenizer=tokenizer,
            suppress_token_ids=suppress_token_ids,
        )
        generation_time = time.perf_counter() - generation_start
        selection_start = time.perf_counter()
        candidate_confidence_records = evaluate_front_candidate_causal_confidence(
            model,
            out,
            prefix_ids=prefix_ids,
            front_mask_lengths=front_mask_lengths,
            suffix_ids=suffix_ids,
            attention_mask=prefix_mask,
            suffix_attention_mask=suffix_mask,
            cfg_scale=args.cfg_scale,
            mask_id=args.mask_id,
            logits_eos_inf=args.logits_eos_inf,
            confidence_eos_eot_inf=args.confidence_eos_eot_inf,
            position_mode=args.position_mode,
            tokenizer=None,
            verbose=False,
            score_basis=args.posthoc_score_basis,
            score_topk=args.posthoc_score_topk,
            suffix_head_length=0,
            suffix_relative_positions=effective_suffix_positions_by_sample,
            full_suffix_context=getattr(args, "posthoc_full_suffix_context", False),
        )
        selection_time += time.perf_counter() - selection_start
    front_batch, _middle_batch = get_mask_regions(
        out=out,
        prefix_ids=prefix_ids,
        front_mask_lengths=front_mask_lengths,
        middle_mask_len=middle_mask_len,
    )
    slot_offsets = []
    cursor = 0
    for slot_length in front_mask_lengths:
        slot_offsets.append((cursor, cursor + slot_length))
        cursor += slot_length
    samples = []
    for idx, problem in enumerate(problems_batch):
        all_front_slot_token_ids = []
        all_front_slot_tokens = []
        all_front_completions = []
        for slot_index, slot_length in enumerate(front_mask_lengths):
            slot_start, slot_end = slot_offsets[slot_index]
            slot_token_ids = (
                front_batch[idx, slot_start:slot_end].detach().cpu().tolist()
            )
            all_front_slot_token_ids.append(slot_token_ids)
            all_front_slot_tokens.append(
                tokenizer.convert_ids_to_tokens(slot_token_ids)
            )
            all_front_completions.append(
                tokenizer.decode(slot_token_ids, skip_special_tokens=True)
            )
        (
            front_posthoc_score_by_slot,
            _front_posthoc_avg_score_by_slot,
            _front_posthoc_min_score_by_slot,
        ) = build_posthoc_score_by_slot(
            candidate_confidence_records, idx, front_mask_lengths
        )
        selected_slot_indices = list(range(len(front_mask_lengths)))
        if False and True and selected_slot_indices:
            if getattr(args, "posthoc_filter_invalid_ast", False):
                for slot_index in selected_slot_indices:
                    slot_start, slot_end = slot_offsets[slot_index]
                    slot_token_ids = (
                        front_batch[idx, slot_start:slot_end].detach().cpu().tolist()
                    )
                    slot_completion = tokenizer.decode(
                        slot_token_ids, skip_special_tokens=True
                    )
                    if candidate_has_valid_python_ast(problem, slot_completion):
                        selected_slot_indices = [slot_index]
                        break
                else:
                    selected_slot_indices = [selected_slot_indices[0]]
            else:
                selected_slot_indices = [selected_slot_indices[0]]
        else:
            selection_start = time.perf_counter()
            score_list = []
            ast_valid_slot_indices = []
            eligible_slot_indices = []
            special_clean_slot_indices = []
            ast_filter_enabled = bool(
                getattr(args, "posthoc_filter_invalid_ast", False)
            )
            for slot_index, slot_length in enumerate(front_mask_lengths):
                token_scores = front_posthoc_score_by_slot[slot_index]
                slot_start, slot_end = slot_offsets[slot_index]
                slot_token_ids = (
                    front_batch[idx, slot_start:slot_end].detach().cpu().tolist()
                )
                if ast_filter_enabled:
                    slot_completion = tokenizer.decode(
                        slot_token_ids, skip_special_tokens=True
                    )
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
                    candidate_pool,
                    key=lambda slot_index: score_list[slot_index],
                    reverse=True,
                )
                topk = min(1, len(ranked_slot_indices))
                selected_slot_indices = [
                    int(slot_index) for slot_index in ranked_slot_indices[:topk]
                ]
            selection_time += time.perf_counter() - selection_start
        for slot_index in selected_slot_indices:
            slot_length = front_mask_lengths[slot_index]
            slot_token_ids = all_front_slot_token_ids[slot_index]
            slot_completion = all_front_completions[slot_index]
            sample = build_sample(
                problem=problem,
                slot_index=slot_index,
                slot_length=slot_length,
                completion=slot_completion,
            )
            sample["posthoc_full_suffix_context"] = bool(
                getattr(args, "posthoc_full_suffix_context", False)
            )
            sample["front_mask_lengths"] = list(front_mask_lengths)
            sample["generated_token_count"] = len(slot_token_ids)
            sample["generated_token_ids"] = slot_token_ids
            sample["generated_tokens"] = all_front_slot_tokens[slot_index]
            sample["all_front_completions"] = list(all_front_completions)
            sample["all_front_slot_token_ids"] = [
                list(token_ids) for token_ids in all_front_slot_token_ids
            ]
            sample["all_front_slot_tokens"] = [
                list(tokens) for tokens in all_front_slot_tokens
            ]
            samples.append(sample)
            if getattr(args, "show_posthoc_suffix_selection", False):
                suffix_selection_details = build_suffix_selection_details(
                    problem=problem,
                    tokenizer=tokenizer,
                    final_positions=effective_suffix_positions_by_sample[idx],
                    sparse_positions=sparse_suffix_positions_by_sample[idx],
                )
                print_suffix_selection_details(
                    problem["task_id"], suffix_selection_details
                )
    add_stage_timing(
        args, generation_seconds=generation_time, selection_seconds=selection_time
    )
    return samples
