import json
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from transformers import AutoConfig, PreTrainedTokenizerFast

from app.config import DEVICE, MODEL_FILE, MODEL_PATH, N_CTX
from app.decoder import (
    VITERBI_BIAS_KEYS,
    ViterbiDecoder,
    build_label_info,
    labels_to_detections,
    select_non_overlapping_detections,
)
from app.schemas import Detection


@dataclass(frozen=True)
class PyTorchTokenClassifier:
    model: Any
    device: Any
    torch: Any

    def logits(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        model_inputs = {
            "input_ids": self.torch.as_tensor(input_ids, dtype=self.torch.long, device=self.device),
            "attention_mask": self.torch.as_tensor(attention_mask, dtype=self.torch.long, device=self.device),
        }

        with self.torch.inference_mode():
            outputs = self.model(**model_inputs)

        return outputs.logits.detach().float().cpu().numpy()


@dataclass(frozen=True)
class PrivacyFilterRuntime:
    model: PyTorchTokenClassifier
    tokenizer: PreTrainedTokenizerFast
    decoder: ViterbiDecoder
    n_ctx: int


_runtime: PrivacyFilterRuntime | None = None


def load_classifier() -> PrivacyFilterRuntime:
    global _runtime

    if _runtime is not None:
        return _runtime

    ensure_model_dir()
    config = AutoConfig.from_pretrained(str(MODEL_PATH))
    tokenizer = load_tokenizer()
    n_ctx = resolve_n_ctx(config)
    label_info = build_label_info(config.id2label)
    decoder = ViterbiDecoder(label_info=label_info, **load_viterbi_biases())
    model = load_pytorch_model()
    _runtime = PrivacyFilterRuntime(model=model, tokenizer=tokenizer, decoder=decoder, n_ctx=n_ctx)
    return _runtime


def load_pytorch_model() -> PyTorchTokenClassifier:
    try:
        import torch
        install_torch_dynamo_stub(torch)
        from transformers import AutoModelForTokenClassification
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load model.safetensors. Install requirements.txt first.") from exc

    device = resolve_device(torch)
    model = AutoModelForTokenClassification.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()
    return PyTorchTokenClassifier(model=model, device=device, torch=torch)


def install_torch_dynamo_stub(torch: Any) -> None:
    if "torch._dynamo" in sys.modules:
        return

    dynamo = ModuleType("torch._dynamo")
    trace_wrapped = ModuleType("torch._dynamo._trace_wrapped_higher_order_op")

    def identity(function: Any) -> Any:
        return function

    def mark_static_address(*_args: Any, **_kwargs: Any) -> None:
        return None

    class TransformGetItemToIndex:
        def __enter__(self) -> "TransformGetItemToIndex":
            return self

        def __exit__(self, *_args: Any) -> bool:
            return False

    dynamo.allow_in_graph = identity
    dynamo.assume_constant_result = identity
    dynamo.mark_static_address = mark_static_address
    trace_wrapped.TransformGetItemToIndex = TransformGetItemToIndex
    dynamo._trace_wrapped_higher_order_op = trace_wrapped

    sys.modules["torch._dynamo"] = dynamo
    sys.modules["torch._dynamo._trace_wrapped_higher_order_op"] = trace_wrapped
    torch._dynamo = dynamo


def resolve_device(torch: Any) -> Any:
    if DEVICE != "auto":
        return torch.device(DEVICE)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer_config_path = MODEL_PATH / "tokenizer_config.json"
    tokenizer_json_path = MODEL_PATH / "tokenizer.json"

    with tokenizer_config_path.open(encoding="utf-8") as file:
        tokenizer_config = json.load(file)

    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json_path),
        eos_token=tokenizer_config.get("eos_token"),
        pad_token=tokenizer_config.get("pad_token"),
        model_max_length=tokenizer_config.get("model_max_length", 128000),
    )


def load_viterbi_biases() -> dict[str, float]:
    calibration_path = MODEL_PATH / "viterbi_calibration.json"
    if not calibration_path.exists():
        return {key: 0.0 for key in VITERBI_BIAS_KEYS}

    with calibration_path.open(encoding="utf-8") as file:
        payload = json.load(file)

    operating_points = payload.get("operating_points", {})
    default_point = operating_points.get("default", {})
    biases = default_point.get("biases", {})
    return {key: float(biases.get(key, 0.0)) for key in VITERBI_BIAS_KEYS}


def resolve_n_ctx(config: Any) -> int:
    if N_CTX is not None:
        try:
            value = int(N_CTX)
        except ValueError:
            raise ValueError("OPF_N_CTX must be a positive integer") from None
        if value <= 0:
            raise ValueError("OPF_N_CTX must be a positive integer")
        return value

    if DEVICE == "cpu":
        return 4096

    for field_name in ("default_n_ctx", "initial_context_length", "max_position_embeddings"):
        value = getattr(config, field_name, None)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Checkpoint config field {field_name} must be a positive integer")
        if value <= 0:
            raise ValueError(f"Checkpoint config field {field_name} must be positive")
        return value

    return 4096


def ensure_model_dir() -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    config_file = MODEL_PATH / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Model config file not found: {config_file}")

    tokenizer_file = MODEL_PATH / "tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

    tokenizer_config_file = MODEL_PATH / "tokenizer_config.json"
    if not tokenizer_config_file.exists():
        raise FileNotFoundError(f"Tokenizer config file not found: {tokenizer_config_file}")

    model_file = MODEL_PATH / MODEL_FILE
    if model_file.exists():
        return

    raise FileNotFoundError(f"PyTorch model file not found: {model_file}")


def get_classifier() -> PrivacyFilterRuntime:
    if _runtime is None:
        raise RuntimeError("classifier is not loaded")
    return _runtime


def is_classifier_loaded() -> bool:
    return _runtime is not None


def run_detection(text: str, threshold: float) -> list[Detection]:
    runtime = get_classifier()
    tokenized = runtime.tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    token_ids = [int(token_id) for token_id in tokenized["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in tokenized["offset_mapping"]]
    if not token_ids:
        return []

    token_logprobs, token_positions = aggregate_token_logprobs(runtime, token_ids)
    if not token_positions:
        return []

    labels = runtime.decoder.decode(token_logprobs)
    if len(labels) != len(token_positions):
        labels = token_logprobs.argmax(axis=1).tolist()

    selected_offsets = [offsets[index] for index in token_positions]
    detections = labels_to_detections(
        text=text,
        labels=labels,
        logprobs=token_logprobs,
        offsets=selected_offsets,
        label_info=runtime.decoder.label_info,
    )
    filtered = [detection for detection in detections if detection.score >= threshold]
    return select_non_overlapping_detections(filtered)


def aggregate_token_logprobs(
    runtime: PrivacyFilterRuntime,
    token_ids: list[int],
) -> tuple[np.ndarray, list[int]]:
    logprob_logsumexp: list[np.ndarray | None] = [None] * len(token_ids)
    counts = [0] * len(token_ids)

    for window_start, window_tokens in iter_windows(token_ids, runtime.n_ctx):
        input_ids = np.asarray([window_tokens], dtype=np.int64)
        attention_mask = np.ones_like(input_ids, dtype=np.int64)
        logits = runtime.model.logits(input_ids=input_ids, attention_mask=attention_mask)[0]
        logprobs = log_softmax(logits.astype(np.float32, copy=False), axis=-1)
        if logprobs.shape[0] != len(window_tokens):
            raise ValueError("Logprob output length does not match window length")

        for token_pos, score_vector in enumerate(logprobs):
            token_index = window_start + token_pos
            existing = logprob_logsumexp[token_index]
            if existing is None:
                logprob_logsumexp[token_index] = score_vector.copy()
            else:
                logprob_logsumexp[token_index] = np.logaddexp(existing, score_vector)
            counts[token_index] += 1

    token_positions: list[int] = []
    token_score_vectors: list[np.ndarray] = []
    for token_index, score_sum in enumerate(logprob_logsumexp):
        count = counts[token_index]
        if score_sum is None or count <= 0:
            continue
        token_positions.append(token_index)
        token_score_vectors.append(score_sum - np.log(float(count)))

    if not token_score_vectors:
        return np.empty((0, 0), dtype=np.float32), []

    return np.stack(token_score_vectors, axis=0), token_positions


def iter_windows(token_ids: list[int], window_size: int) -> Iterator[tuple[int, list[int]]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    for start in range(0, len(token_ids), window_size):
        end = min(start + window_size, len(token_ids))
        yield start, token_ids[start:end]


def log_softmax(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_values
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def redact_text(text: str, detections: list[Detection], mask: str | None) -> str:
    if not detections:
        return text

    chunks: list[str] = []
    cursor = 0
    for detection in detections:
        if detection.start < cursor:
            continue
        chunks.append(text[cursor : detection.start])
        chunks.append(build_replacement(detection.text, detection.label, mask))
        cursor = detection.end

    chunks.append(text[cursor:])
    return "".join(chunks)


def build_replacement(value: str, label: str, mask: str | None) -> str:
    if not value.strip():
        return value

    replacement = mask or f"[{label}]"
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    return f"{leading}{replacement}{trailing}"
