"""Command-line entry point for PILL infilling."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="PILL infilling with a pretrained length probe."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="JSONL with prefix and suffix strings.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--model-path", help="Local copy of the checkpoint's exact base model."
    )
    parser.add_argument(
        "--probe-feature-batch-size",
        type=int,
        default=1,
        help="Probe-feature batch size; use 4 for the original Human-S setup.",
    )
    parser.add_argument("--candidate-radius", type=int, default=2)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--suffix-head-length", type=int)
    parser.add_argument("--syntax-filter", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.probe_feature_batch_size < 1:
        parser.error("--probe-feature-batch-size must be positive.")
    if args.input.resolve() == args.output.resolve():
        parser.error("Input and output must be different files.")
    records = [
        json.loads(line) for line in args.input.read_text().splitlines() if line.strip()
    ]
    if not records:
        parser.error("Input must contain at least one record.")
    for record in records:
        if not isinstance(record, dict) or not all(
            isinstance(record.get(k), str) for k in ("prefix", "suffix")
        ):
            parser.error("Each record needs string prefix and suffix fields.")
    import torch
    from pill.inference import PILL

    torch.manual_seed(args.seed)
    engine = PILL(args.checkpoint, args.device, args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for result in engine.infill_many(
            records,
            probe_feature_batch_size=args.probe_feature_batch_size,
            candidate_radius=args.candidate_radius,
            max_length=args.max_length,
            steps=args.steps,
            temperature=args.temperature,
            syntax_filter=args.syntax_filter,
            alpha=args.alpha,
            suffix_head_length=args.suffix_head_length,
        ):
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()


if __name__ == "__main__":
    main()
