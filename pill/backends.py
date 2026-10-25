import torch
from transformers import AutoTokenizer
from .models.modeling_llada import LLaDAModelLM
from .models.modeling_dream import DreamModel
from .models.modeling_lladamoe import LLaDAMoEModelLM

DEFAULT_BACKEND_MASK_IDS = {"llada": 126336, "llada_moe": 156895}


def infer_model_backend(
    model_name: str | None, model_backend: str | None = None
) -> str:
    if model_backend in {"llada", "llada_moe", "dream"}:
        return model_backend
    normalized_name = (model_name or "").lower()
    if "dream" in normalized_name:
        return "dream"
    if (
        "llada-moe" in normalized_name
        or "lladamoe" in normalized_name
        or "a1b" in normalized_name
    ):
        return "llada_moe"
    return "llada"


def load_model_and_tokenizer(
    model_name: str,
    device: str,
    torch_dtype: torch.dtype,
    model_backend: str = "auto",
):
    backend = infer_model_backend(model_name=model_name, model_backend=model_backend)
    model_cls = (
        LLaDAMoEModelLM
        if backend == "llada_moe"
        else (DreamModel if backend == "dream" else LLaDAModelLM)
    )
    model_kwargs = {
        "torch_dtype": torch_dtype,
    }
    if backend == "llada_moe":
        model_kwargs["attn_implementation"] = "eager"
    model = (
        model_cls.from_pretrained(
            model_name,
            **model_kwargs,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    return model, tokenizer, backend


def resolve_runtime_mask_id(
    requested_mask_id: int | None,
    model,
    tokenizer=None,
    model_backend: str | None = None,
) -> int:
    if requested_mask_id is not None:
        return int(requested_mask_id)
    model_mask_id = getattr(model.config, "mask_token_id", None)
    if model_mask_id is not None:
        return int(model_mask_id)
    tokenizer_mask_id = (
        getattr(tokenizer, "mask_token_id", None) if tokenizer is not None else None
    )
    if tokenizer_mask_id is not None:
        return int(tokenizer_mask_id)
    backend = infer_model_backend(
        getattr(model.config, "_name_or_path", None), model_backend
    )
    backend_mask_id = DEFAULT_BACKEND_MASK_IDS.get(backend)
    if backend_mask_id is not None:
        return backend_mask_id
    raise ValueError(
        "mask_id was not provided and neither model nor tokenizer exposes a mask token id."
    )
