"""CPU tests exercise actual backbone implementations with tiny random models."""

import unittest
from types import SimpleNamespace
import torch
from pill.defaults import DEFAULTS
from pill.inference import PILL, candidate_lengths, predict_length, predict_lengths
from pill.probe import MLPRegressor
from pill.selection import run_generation_batch
from pill.dream import run_generation_batch_dream
from pill.models.configuration_llada import LLaDAConfig
from pill.models.modeling_llada import LLaDAModelLM
from pill.models.configuration_dream import DreamConfig
from pill.models.modeling_dream import DreamModel
from pill.models.configuration_lladamoe import LLaDAConfig as MoEConfig
from pill.models.modeling_lladamoe import LLaDAMoEModelLM


class TinyTokenizer:
    pad_token_id = 0
    mask_token_id = 1
    all_special_ids = [0, 1]

    def __call__(self, texts, padding=False, return_tensors=None, **kwargs):
        single = isinstance(texts, str)
        rows = [
            [2 + ord(c) % 29 for c in text] for text in ([texts] if single else texts)
        ]
        if return_tensors is None:
            return {"input_ids": rows[0] if single else rows}
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor(
                [[0] * (width - len(r)) + r for r in rows], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [[0] * (width - len(r)) + [1] * len(r) for r in rows], dtype=torch.long
            ),
        }

    def decode(self, ids, **kwargs):
        return "".join(chr(97 + int(i) % 26) for i in ids)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self.decode([ids])
        return [self.decode([i]) for i in ids]


def tiny_models():
    yield (
        "llada",
        LLaDAModelLM(
            LLaDAConfig(
                d_model=16,
                n_heads=2,
                n_layers=2,
                vocab_size=32,
                embedding_size=32,
                max_sequence_length=128,
                rope=True,
                weight_tying=False,
                eos_token_id=0,
                pad_token_id=0,
            )
        ).eval(),
    )
    yield (
        "dream",
        DreamModel(
            DreamConfig(
                vocab_size=32,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=128,
                mask_token_id=1,
                pad_token_id=0,
            )
        ).eval(),
    )
    yield (
        "llada_moe",
        LLaDAMoEModelLM(
            MoEConfig(
                vocab_size=32,
                hidden_size=16,
                dense_intermediate_size=32,
                expert_intermediate_size=16,
                shared_expert_intermediate_size=16,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=2,
                rope_theta=10000,
                partial_rotary_factor=1.0,
                num_experts=2,
                num_experts_per_tok=1,
                moe_layer_freq=[1, 1],
                norm_topk_prob=True,
                qk_layernorm=False,
                moe_router_enable_expert_bias=False,
                moe_router_score_function="softmax",
                router_num_group=1,
                router_topk_group=1,
                pad_token_id=0,
                eos_token_id=0,
                attn_implementation="eager",
            )
        ).eval(),
    )


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_candidate_boundaries(self):
        self.assertEqual(candidate_lengths(0.1), (1, [1, 2, 3]))
        self.assertEqual(candidate_lengths(50, max_length=4), (4, [2, 3, 4]))
        with self.assertRaises(ValueError):
            candidate_lengths(3, radius=-1)

    def test_batched_public_api(self):
        records = [
            {"task_id": "first", "prefix": "ab", "suffix": "cdefg"},
            {"task_id": "second", "prefix": "", "suffix": "cd"},
        ]
        for backend, model in tiny_models():
            with self.subTest(backend=backend):
                engine = PILL.__new__(PILL)
                engine.model, engine.tokenizer = model, TinyTokenizer()
                engine.device, engine.backend, engine.mask_id = "cpu", backend, 1
                engine.probe = MLPRegressor(48, [8], 0, "zero", True).eval()
                engine.payload = {
                    "model_backend": backend,
                    "model_name": backend,
                    "task": "python-single",
                    "layer": {
                        "hidden_state_index": 2,
                        "probe_config": {
                            "use_context_features": True,
                            "context_pool_size": 4,
                        },
                        "target_mean": 1.0,
                        "target_std": 0.2,
                    },
                }
                direct = [
                    engine.infill(
                        row["prefix"],
                        row["suffix"],
                        task_id=row["task_id"],
                        max_length=3,
                    )
                    for row in records
                ]
                sequential = list(
                    engine.infill_many(
                        records, probe_feature_batch_size=1, max_length=3
                    )
                )
                self.assertEqual(direct, sequential)
                batched = list(
                    engine.infill_many(
                        records, probe_feature_batch_size=2, max_length=3
                    )
                )
                predictions = predict_lengths(
                    model,
                    engine.tokenizer,
                    engine.probe,
                    engine.payload,
                    [
                        {"prompt": row["prefix"], "suffix": row["suffix"]}
                        for row in records
                    ],
                    1,
                    "cpu",
                )
                self.assertEqual(
                    [row["task_id"] for row in batched], ["first", "second"]
                )
                self.assertEqual([row["prediction"] for row in batched], predictions)
                with self.assertRaises(ValueError):
                    list(engine.infill_many(records, probe_feature_batch_size=0))

    def test_all_backends(self):
        tokenizer = TinyTokenizer()
        for backend, model in tiny_models():
            with self.subTest(backend=backend), torch.inference_mode():
                probe = MLPRegressor(48, [8], 0, "zero", True).eval()
                payload = {
                    "model_backend": backend,
                    "layer": {
                        "hidden_state_index": 2,
                        "probe_config": {
                            "use_context_features": True,
                            "context_pool_size": 4,
                        },
                        "target_mean": 1.0,
                        "target_std": 0.2,
                    },
                }
                for prefix, suffix in [("ab", "cdefg"), ("", "cde"), ("ab", "")]:
                    prediction = predict_length(
                        model, tokenizer, probe, payload, prefix, suffix, 1, "cpu"
                    )
                    self.assertGreaterEqual(prediction, 0)
                args = SimpleNamespace(**DEFAULTS)
                args.mask_id = 1
                args.middle_mask_len = 2
                args.front_mask_lengths = [1, 2, 3]
                args.steps = 3
                problems = [{"task_id": "tiny", "prompt": "ab", "suffix": "cdefg"}]
                if backend == "dream":
                    samples = run_generation_batch_dream(
                        model, tokenizer, "cpu", problems, 2, [1, 2, 3], args
                    )
                else:
                    samples = run_generation_batch(
                        model, tokenizer, "cpu", problems, 2, args
                    )
                self.assertEqual(len(samples), 1)
                self.assertEqual(len(samples[0]["all_front_completions"]), 3)
                self.assertIn(samples[0]["mask_token_length"], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
