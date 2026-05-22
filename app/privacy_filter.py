import json
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
from transformers import AutoConfig, PreTrainedTokenizerFast

from app.config import DECODER, DEVICE, INFERENCE_BATCH_SIZE, MODEL_FILE, MODEL_PATH, N_CTX, PROFILE
from app.decoder import (
    VITERBI_BIAS_KEYS,
    ViterbiDecoder,
    build_label_info,
    labels_to_detections,
    select_non_overlapping_detections,
)
from app.schemas import Detection


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class PyTorchTokenClassifier:
    model: Any
    device: Any
    torch: Any

    def logits_tensor(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> Any:
        model_inputs = {
            "input_ids": self.torch.as_tensor(input_ids, dtype=self.torch.long, device=self.device),
            "attention_mask": self.torch.as_tensor(attention_mask, dtype=self.torch.long, device=self.device),
        }

        with self.torch.inference_mode():
            outputs = self.model(**model_inputs)

        return outputs.logits.detach().float()

    def logits(self, input_ids: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        return self.logits_tensor(input_ids=input_ids, attention_mask=attention_mask).cpu().numpy()


@dataclass(frozen=True)
class PrivacyFilterRuntime:
    model: PyTorchTokenClassifier
    tokenizer: PreTrainedTokenizerFast
    decoder: ViterbiDecoder
    n_ctx: int
    inference_batch_size: int
    decoder_mode: str


_runtime: PrivacyFilterRuntime | None = None


def load_classifier() -> PrivacyFilterRuntime:
    global _runtime

    if _runtime is not None:
        return _runtime

    ensure_model_dir()
    config = AutoConfig.from_pretrained(str(MODEL_PATH))
    tokenizer = load_tokenizer()
    n_ctx = resolve_n_ctx(config)
    inference_batch_size = resolve_inference_batch_size()
    decoder_mode = resolve_decoder_mode()
    label_info = build_label_info(config.id2label)
    decoder = ViterbiDecoder(label_info=label_info, **load_viterbi_biases())
    model = load_pytorch_model()
    _runtime = PrivacyFilterRuntime(
        model=model,
        tokenizer=tokenizer,
        decoder=decoder,
        n_ctx=n_ctx,
        inference_batch_size=inference_batch_size,
        decoder_mode=decoder_mode,
    )
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

    def disable(function: Any = None, *_args: Any, **_kwargs: Any) -> Any:
        if function is None:
            return identity
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
    dynamo.disable = disable
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


def resolve_inference_batch_size() -> int:
    try:
        value = int(INFERENCE_BATCH_SIZE)
    except ValueError:
        raise ValueError("OPF_INFERENCE_BATCH_SIZE must be a positive integer") from None

    if value <= 0:
        raise ValueError("OPF_INFERENCE_BATCH_SIZE must be a positive integer")
    return value


def resolve_decoder_mode() -> str:
    mode = DECODER.strip().lower()
    if mode in {"viterbi", "argmax"}:
        return mode
    raise ValueError("OPF_DECODER must be viterbi or argmax")


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
    total_started_at = time.perf_counter()
    runtime = get_classifier()

    started_at = time.perf_counter()
    tokenized = runtime.tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    tokenize_ms = elapsed_ms(started_at)

    started_at = time.perf_counter()
    token_ids = [int(token_id) for token_id in tokenized["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in tokenized["offset_mapping"]]
    prepare_ms = elapsed_ms(started_at)
    if not token_ids:
        log_detection_profile(
            text=text,
            tokens=0,
            detections=0,
            total_ms=elapsed_ms(total_started_at),
            tokenize_ms=tokenize_ms,
            prepare_ms=prepare_ms,
            inference_stats={},
            decode_ms=0.0,
            spans_ms=0.0,
        )
        return []

    token_logprobs, token_positions, inference_stats = aggregate_token_logprobs(runtime, token_ids)
    if not token_positions:
        log_detection_profile(
            text=text,
            tokens=len(token_ids),
            detections=0,
            total_ms=elapsed_ms(total_started_at),
            tokenize_ms=tokenize_ms,
            prepare_ms=prepare_ms,
            inference_stats=inference_stats,
            decode_ms=0.0,
            spans_ms=0.0,
        )
        return []

    started_at = time.perf_counter()
    labels = decode_labels(runtime, token_logprobs, token_positions)
    decode_ms = elapsed_ms(started_at)

    started_at = time.perf_counter()
    selected_offsets = [offsets[index] for index in token_positions]
    detection_logprobs = numpy_logprobs(token_logprobs)
    detections = labels_to_detections(
        text=text,
        labels=labels,
        logprobs=detection_logprobs,
        offsets=selected_offsets,
        label_info=runtime.decoder.label_info,
    )
    filtered = [detection for detection in detections if detection.score >= threshold]
    selected = select_non_overlapping_detections(filtered)
    spans_ms = elapsed_ms(started_at)
    log_detection_profile(
        text=text,
        tokens=len(token_ids),
        detections=len(selected),
        total_ms=elapsed_ms(total_started_at),
        tokenize_ms=tokenize_ms,
        prepare_ms=prepare_ms,
        inference_stats=inference_stats,
        decode_ms=decode_ms,
        spans_ms=spans_ms,
    )
    return selected


def decode_labels(
    runtime: PrivacyFilterRuntime,
    token_logprobs: Any,
    token_positions: list[int],
) -> list[int]:
    if runtime.decoder_mode == "argmax":
        if isinstance(token_logprobs, np.ndarray):
            return token_logprobs.argmax(axis=1).tolist()
        return token_logprobs.argmax(dim=1).detach().cpu().tolist()

    if should_decode_viterbi_on_device(runtime, token_logprobs):
        labels = runtime.decoder.decode_torch(token_logprobs)
        if len(labels) == len(token_positions):
            return labels
        return token_logprobs.argmax(dim=1).detach().cpu().tolist()

    labels = runtime.decoder.decode(token_logprobs)
    if len(labels) == len(token_positions):
        return labels
    return token_logprobs.argmax(axis=1).tolist()


def aggregate_token_logprobs(
    runtime: PrivacyFilterRuntime,
    token_ids: list[int],
) -> tuple[Any, list[int], dict[str, float | int]]:
    if should_keep_logprobs_on_device(runtime):
        return aggregate_token_logprobs_torch(runtime, token_ids)

    logprob_logsumexp: list[np.ndarray | None] = [None] * len(token_ids)
    counts = [0] * len(token_ids)
    stats: dict[str, float | int] = {
        "batches": 0,
        "windows": 0,
        "model_ms": 0.0,
        "logprob_ms": 0.0,
        "collect_ms": 0.0,
        "finalize_ms": 0.0,
    }

    windows = iter_windows(token_ids, runtime.n_ctx)
    for window_batch in iter_batches(windows, runtime.inference_batch_size):
        input_ids, attention_mask = build_batch_inputs(window_batch)
        stats["batches"] = int(stats["batches"]) + 1
        stats["windows"] = int(stats["windows"]) + len(window_batch)

        started_at = time.perf_counter()
        batch_logits = runtime.model.logits(input_ids=input_ids, attention_mask=attention_mask)
        stats["model_ms"] = float(stats["model_ms"]) + elapsed_ms(started_at)

        for batch_index, (window_start, window_tokens) in enumerate(window_batch):
            logits = batch_logits[batch_index, : len(window_tokens)]
            started_at = time.perf_counter()
            logprobs = log_softmax(logits.astype(np.float32, copy=False), axis=-1)
            stats["logprob_ms"] = float(stats["logprob_ms"]) + elapsed_ms(started_at)
            if logprobs.shape[0] != len(window_tokens):
                raise ValueError("Logprob output length does not match window length")

            started_at = time.perf_counter()
            for token_pos, score_vector in enumerate(logprobs):
                token_index = window_start + token_pos
                existing = logprob_logsumexp[token_index]
                if existing is None:
                    logprob_logsumexp[token_index] = score_vector.copy()
                else:
                    logprob_logsumexp[token_index] = np.logaddexp(existing, score_vector)
                counts[token_index] += 1
            stats["collect_ms"] = float(stats["collect_ms"]) + elapsed_ms(started_at)

    started_at = time.perf_counter()
    token_positions: list[int] = []
    token_score_vectors: list[np.ndarray] = []
    for token_index, score_sum in enumerate(logprob_logsumexp):
        count = counts[token_index]
        if score_sum is None or count <= 0:
            continue
        token_positions.append(token_index)
        token_score_vectors.append(score_sum - np.log(float(count)))

    if not token_score_vectors:
        stats["finalize_ms"] = float(stats["finalize_ms"]) + elapsed_ms(started_at)
        return np.empty((0, 0), dtype=np.float32), [], stats

    output = np.stack(token_score_vectors, axis=0)
    stats["finalize_ms"] = float(stats["finalize_ms"]) + elapsed_ms(started_at)
    return output, token_positions, stats


def aggregate_token_logprobs_torch(
    runtime: PrivacyFilterRuntime,
    token_ids: list[int],
) -> tuple[Any, list[int], dict[str, float | int]]:
    token_logprobs: list[Any] = []
    token_positions: list[int] = []
    stats: dict[str, float | int] = {
        "batches": 0,
        "windows": 0,
        "model_ms": 0.0,
        "logprob_ms": 0.0,
        "collect_ms": 0.0,
        "finalize_ms": 0.0,
    }

    windows = iter_windows(token_ids, runtime.n_ctx)
    for window_batch in iter_batches(windows, runtime.inference_batch_size):
        input_ids, attention_mask = build_batch_inputs(window_batch)
        stats["batches"] = int(stats["batches"]) + 1
        stats["windows"] = int(stats["windows"]) + len(window_batch)

        started_at = time.perf_counter()
        batch_logits = runtime.model.logits_tensor(input_ids=input_ids, attention_mask=attention_mask)
        synchronize_for_profile(runtime)
        stats["model_ms"] = float(stats["model_ms"]) + elapsed_ms(started_at)

        started_at = time.perf_counter()
        batch_logprobs = runtime.model.torch.nn.functional.log_softmax(batch_logits, dim=-1)
        synchronize_for_profile(runtime)
        stats["logprob_ms"] = float(stats["logprob_ms"]) + elapsed_ms(started_at)

        started_at = time.perf_counter()
        for batch_index, (window_start, window_tokens) in enumerate(window_batch):
            token_logprobs.append(batch_logprobs[batch_index, : len(window_tokens)])
            token_positions.extend(range(window_start, window_start + len(window_tokens)))
        stats["collect_ms"] = float(stats["collect_ms"]) + elapsed_ms(started_at)

    started_at = time.perf_counter()
    if not token_logprobs:
        stats["finalize_ms"] = float(stats["finalize_ms"]) + elapsed_ms(started_at)
        empty = runtime.model.torch.empty((0, 0), device=runtime.model.device, dtype=runtime.model.torch.float32)
        return empty, [], stats

    output = runtime.model.torch.cat(token_logprobs, dim=0)
    synchronize_for_profile(runtime)
    stats["finalize_ms"] = float(stats["finalize_ms"]) + elapsed_ms(started_at)
    return output, token_positions, stats


def should_keep_logprobs_on_device(runtime: PrivacyFilterRuntime) -> bool:
    return runtime.decoder_mode == "viterbi" and runtime.model.device.type == "cuda"


def should_decode_viterbi_on_device(runtime: PrivacyFilterRuntime, token_logprobs: Any) -> bool:
    if runtime.decoder_mode != "viterbi":
        return False
    if isinstance(token_logprobs, np.ndarray):
        return False
    return token_logprobs.device.type == "cuda"


def numpy_logprobs(token_logprobs: Any) -> np.ndarray:
    if isinstance(token_logprobs, np.ndarray):
        return token_logprobs
    return token_logprobs.detach().float().cpu().numpy()


def synchronize_for_profile(runtime: PrivacyFilterRuntime) -> None:
    if not PROFILE:
        return
    if runtime.model.device.type != "cuda":
        return
    runtime.model.torch.cuda.synchronize(runtime.model.device)


def iter_windows(token_ids: list[int], window_size: int) -> Iterator[tuple[int, list[int]]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    for start in range(0, len(token_ids), window_size):
        end = min(start + window_size, len(token_ids))
        yield start, token_ids[start:end]


def iter_batches(
    windows: Iterator[tuple[int, list[int]]],
    batch_size: int,
) -> Iterator[list[tuple[int, list[int]]]]:
    batch: list[tuple[int, list[int]]] = []
    for window in windows:
        batch.append(window)
        if len(batch) < batch_size:
            continue
        yield batch
        batch = []

    if not batch:
        return
    yield batch


def build_batch_inputs(window_batch: list[tuple[int, list[int]]]) -> tuple[np.ndarray, np.ndarray]:
    max_length = max(len(window_tokens) for _, window_tokens in window_batch)
    input_ids = np.zeros((len(window_batch), max_length), dtype=np.int64)
    attention_mask = np.zeros_like(input_ids, dtype=np.int64)

    for batch_index, (_, window_tokens) in enumerate(window_batch):
        length = len(window_tokens)
        input_ids[batch_index, :length] = window_tokens
        attention_mask[batch_index, :length] = 1

    return input_ids, attention_mask


def log_softmax(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    shifted = values - max_values
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))


def elapsed_ms(started_at: float) -> float:
    return (time.perf_counter() - started_at) * 1000.0


def log_detection_profile(
    text: str,
    tokens: int,
    detections: int,
    total_ms: float,
    tokenize_ms: float,
    prepare_ms: float,
    inference_stats: dict[str, float | int],
    decode_ms: float,
    spans_ms: float,
) -> None:
    if not PROFILE:
        return

    logger.info(
        "opf_profile chars=%d tokens=%d detections=%d batches=%d windows=%d total_ms=%.2f "
        "tokenize_ms=%.2f prepare_ms=%.2f model_ms=%.2f logprob_ms=%.2f collect_ms=%.2f "
        "finalize_ms=%.2f decode_ms=%.2f spans_ms=%.2f",
        len(text),
        tokens,
        detections,
        int(inference_stats.get("batches", 0)),
        int(inference_stats.get("windows", 0)),
        total_ms,
        tokenize_ms,
        prepare_ms,
        float(inference_stats.get("model_ms", 0.0)),
        float(inference_stats.get("logprob_ms", 0.0)),
        float(inference_stats.get("collect_ms", 0.0)),
        float(inference_stats.get("finalize_ms", 0.0)),
        decode_ms,
        spans_ms,
    )


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
