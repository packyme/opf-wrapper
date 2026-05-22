from dataclasses import dataclass

import numpy as np

from app.schemas import Detection


NEG_INF = -1e9
BACKGROUND_LABEL = "O"
BOUNDARY_PREFIXES = ("B", "I", "E", "S")
VITERBI_BIAS_KEYS = (
    "transition_bias_background_stay",
    "transition_bias_background_to_start",
    "transition_bias_inside_to_continue",
    "transition_bias_inside_to_end",
    "transition_bias_end_to_background",
    "transition_bias_end_to_start",
)


@dataclass(frozen=True)
class LabelInfo:
    token_to_span_label: dict[int, int]
    token_boundary_tags: dict[int, str | None]
    span_class_names: tuple[str, ...]
    background_token_label: int
    background_span_label: int


@dataclass
class ViterbiDecoder:
    label_info: LabelInfo
    transition_bias_background_stay: float = 0.0
    transition_bias_background_to_start: float = 0.0
    transition_bias_inside_to_continue: float = 0.0
    transition_bias_inside_to_end: float = 0.0
    transition_bias_end_to_background: float = 0.0
    transition_bias_end_to_start: float = 0.0

    def __post_init__(self) -> None:
        num_classes = len(self.label_info.token_to_span_label)
        self.start_scores = np.full((num_classes,), NEG_INF, dtype=np.float32)
        self.end_scores = np.full((num_classes,), NEG_INF, dtype=np.float32)
        self.transition_scores = np.full((num_classes, num_classes), NEG_INF, dtype=np.float32)

        for label_id in range(num_classes):
            tag = self.label_info.token_boundary_tags.get(label_id)
            span = self.label_info.token_to_span_label.get(label_id)
            if self.can_start(label_id, tag):
                self.start_scores[label_id] = 0.0
            if self.can_end(label_id, tag):
                self.end_scores[label_id] = 0.0

            for next_label_id in range(num_classes):
                next_tag = self.label_info.token_boundary_tags.get(next_label_id)
                next_span = self.label_info.token_to_span_label.get(next_label_id)
                if not self.is_valid_transition(label_id, tag, span, next_label_id, next_tag, next_span):
                    continue
                self.transition_scores[label_id, next_label_id] = self.transition_bias(
                    label_id,
                    tag,
                    span,
                    next_label_id,
                    next_tag,
                    next_span,
                )

    def can_start(self, label_id: int, tag: str | None) -> bool:
        if label_id == self.label_info.background_token_label:
            return True
        return tag in {"B", "S"}

    def can_end(self, label_id: int, tag: str | None) -> bool:
        if label_id == self.label_info.background_token_label:
            return True
        return tag in {"E", "S"}

    def is_valid_transition(
        self,
        label_id: int,
        tag: str | None,
        span: int | None,
        next_label_id: int,
        next_tag: str | None,
        next_span: int | None,
    ) -> bool:
        next_is_background = self.is_background(next_label_id, next_span)
        if (next_span is None or next_tag is None) and not next_is_background:
            return False
        if self.is_background(label_id, span):
            return next_is_background or next_tag in {"B", "S"}
        if tag in {"E", "S"}:
            return next_is_background or next_tag in {"B", "S"}
        if tag in {"B", "I"}:
            return span == next_span and next_tag in {"I", "E"}
        return False

    def transition_bias(
        self,
        label_id: int,
        tag: str | None,
        span: int | None,
        next_label_id: int,
        next_tag: str | None,
        next_span: int | None,
    ) -> float:
        if self.is_background(label_id, span):
            if self.is_background(next_label_id, next_span):
                return self.transition_bias_background_stay
            if next_tag in {"B", "S"}:
                return self.transition_bias_background_to_start
            return 0.0

        if tag in {"B", "I"}:
            if next_tag == "I" and span == next_span:
                return self.transition_bias_inside_to_continue
            if next_tag == "E" and span == next_span:
                return self.transition_bias_inside_to_end
            return 0.0

        if tag in {"E", "S"}:
            if self.is_background(next_label_id, next_span):
                return self.transition_bias_end_to_background
            if next_tag in {"B", "S"}:
                return self.transition_bias_end_to_start
            return 0.0

        return 0.0

    def is_background(self, label_id: int, span: int | None) -> bool:
        if label_id == self.label_info.background_token_label:
            return True
        return span == self.label_info.background_span_label

    def decode(self, token_logprobs: np.ndarray) -> list[int]:
        if token_logprobs.ndim != 2:
            raise ValueError("token_logprobs must have shape [seq_len, num_classes]")
        if token_logprobs.shape[0] == 0:
            return []

        start_scores = self.start_scores.astype(token_logprobs.dtype, copy=False)
        end_scores = self.end_scores.astype(token_logprobs.dtype, copy=False)
        transition_scores = self.transition_scores.astype(token_logprobs.dtype, copy=False)

        scores = token_logprobs[0] + start_scores
        backpointers = np.empty((token_logprobs.shape[0] - 1, token_logprobs.shape[1]), dtype=np.int64)
        for index in range(1, token_logprobs.shape[0]):
            transitions = scores[:, None] + transition_scores
            best_paths = np.argmax(transitions, axis=0)
            best_scores = transitions[best_paths, np.arange(token_logprobs.shape[1])]
            scores = best_scores + token_logprobs[index]
            backpointers[index - 1] = best_paths

        if not np.isfinite(scores).any():
            return token_logprobs.argmax(axis=1).tolist()

        scores = scores + end_scores
        label = int(scores.argmax())
        path = np.empty((token_logprobs.shape[0],), dtype=np.int64)
        path[-1] = label
        for index in range(token_logprobs.shape[0] - 2, -1, -1):
            label = int(backpointers[index, label])
            path[index] = label
        return path.tolist()

    def decode_torch(self, token_logprobs: object) -> list[int]:
        import torch

        if token_logprobs.ndim != 2:
            raise ValueError("token_logprobs must have shape [seq_len, num_classes]")
        if token_logprobs.shape[0] == 0:
            return []

        device = token_logprobs.device
        dtype = token_logprobs.dtype
        start_scores = torch.as_tensor(self.start_scores, device=device, dtype=dtype)
        end_scores = torch.as_tensor(self.end_scores, device=device, dtype=dtype)
        transition_scores = torch.as_tensor(self.transition_scores, device=device, dtype=dtype)

        scores = token_logprobs[0] + start_scores
        backpointer_dtype = torch.int16 if token_logprobs.shape[1] <= 32767 else torch.int32
        backpointers = torch.empty(
            (token_logprobs.shape[0] - 1, token_logprobs.shape[1]),
            device=device,
            dtype=backpointer_dtype,
        )
        for index in range(1, token_logprobs.shape[0]):
            transitions = scores[:, None] + transition_scores
            best_scores, best_paths = transitions.max(dim=0)
            scores = best_scores + token_logprobs[index]
            backpointers[index - 1] = best_paths.to(backpointer_dtype)

        if not bool(torch.isfinite(scores).any().item()):
            return token_logprobs.argmax(dim=1).detach().cpu().tolist()

        scores = scores + end_scores
        label = scores.argmax()
        path = torch.empty((token_logprobs.shape[0],), device=device, dtype=torch.int64)
        path[-1] = label
        for index in range(token_logprobs.shape[0] - 2, -1, -1):
            label = backpointers[index, label].to(torch.long)
            path[index] = label
        return path.detach().cpu().tolist()


def build_label_info(id2label: dict[int | str, str]) -> LabelInfo:
    names = [id2label[key] for key in sorted(id2label, key=lambda value: int(value))]
    span_class_names = [BACKGROUND_LABEL]
    span_label_lookup = {BACKGROUND_LABEL: 0}
    token_to_span_label: dict[int, int] = {}
    token_boundary_tags: dict[int, str | None] = {}
    background_token_label = None

    for label_id, name in enumerate(names):
        if name == BACKGROUND_LABEL:
            background_token_label = label_id
            token_to_span_label[label_id] = span_label_lookup[BACKGROUND_LABEL]
            token_boundary_tags[label_id] = None
            continue

        boundary, span_name = name.split("-", 1)
        if boundary not in BOUNDARY_PREFIXES:
            raise ValueError(f"Unsupported boundary label: {name}")

        span_id = span_label_lookup.get(span_name)
        if span_id is None:
            span_id = len(span_class_names)
            span_class_names.append(span_name)
            span_label_lookup[span_name] = span_id

        token_to_span_label[label_id] = span_id
        token_boundary_tags[label_id] = boundary

    if background_token_label is None:
        raise ValueError("Label space must include background label O")

    return LabelInfo(
        token_to_span_label=token_to_span_label,
        token_boundary_tags=token_boundary_tags,
        span_class_names=tuple(span_class_names),
        background_token_label=background_token_label,
        background_span_label=span_label_lookup[BACKGROUND_LABEL],
    )


def labels_to_detections(
    text: str,
    labels: list[int],
    logprobs: np.ndarray,
    offsets: list[tuple[int, int]],
    label_info: LabelInfo,
) -> list[Detection]:
    token_spans = labels_to_token_spans(labels, label_info)
    detections: list[Detection] = []

    for span_label, token_start, token_end in token_spans:
        start = offsets[token_start][0]
        end = offsets[token_end - 1][1]
        start, end = trim_span(text, start, end)
        if start >= end:
            continue

        label = label_info.span_class_names[span_label]
        detections.append(
            Detection(
                label=label,
                text=text[start:end],
                score=span_score(labels, logprobs, token_start, token_end),
                start=start,
                end=end,
            )
        )

    return detections


def select_non_overlapping_detections(detections: list[Detection]) -> list[Detection]:
    ordered = sorted(
        detections,
        key=lambda detection: (
            detection.start,
            -(detection.end - detection.start),
            detection.label,
        ),
    )
    kept: list[Detection] = []
    cursor = 0

    for detection in ordered:
        if detection.start < cursor:
            continue
        if detection.end <= detection.start:
            continue
        kept.append(detection)
        cursor = detection.end

    return kept


def labels_to_token_spans(labels: list[int], label_info: LabelInfo) -> list[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    current_label: int | None = None
    start_index: int | None = None
    previous_index: int | None = None

    for index, label_id in enumerate(labels):
        span_label = label_info.token_to_span_label.get(label_id)
        boundary = label_info.token_boundary_tags.get(label_id)

        if previous_index is not None and index != previous_index + 1:
            close_span(spans, current_label, start_index, previous_index + 1)
            current_label = None
            start_index = None

        if span_label is None:
            previous_index = index
            continue

        is_background = span_label == label_info.background_span_label
        if is_background:
            close_span(spans, current_label, start_index, index)
            current_label = None
            start_index = None
            previous_index = index
            continue

        if boundary == "S":
            close_span_with_previous(spans, current_label, start_index, previous_index)
            spans.append((span_label, index, index + 1))
            current_label = None
            start_index = None
        elif boundary == "B":
            close_span_with_previous(spans, current_label, start_index, previous_index)
            current_label = span_label
            start_index = index
        elif boundary == "I":
            if current_label is None or current_label != span_label:
                close_span_with_previous(spans, current_label, start_index, previous_index)
                current_label = span_label
                start_index = index
        elif boundary == "E":
            if current_label is None or current_label != span_label or start_index is None:
                close_span_with_previous(spans, current_label, start_index, previous_index)
                spans.append((span_label, index, index + 1))
                current_label = None
                start_index = None
            else:
                spans.append((current_label, start_index, index + 1))
                current_label = None
                start_index = None
        else:
            close_span_with_previous(spans, current_label, start_index, previous_index)
            current_label = None
            start_index = None

        previous_index = index

    if previous_index is not None:
        close_span(spans, current_label, start_index, previous_index + 1)
    return spans


def close_span_with_previous(
    spans: list[tuple[int, int, int]],
    label: int | None,
    start: int | None,
    previous: int | None,
) -> None:
    if previous is None:
        return
    close_span(spans, label, start, previous + 1)


def close_span(
    spans: list[tuple[int, int, int]],
    label: int | None,
    start: int | None,
    end: int,
) -> None:
    if label is None or start is None or start >= end:
        return
    spans.append((label, start, end))


def trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def span_score(labels: list[int], logprobs: np.ndarray, start: int, end: int) -> float:
    scores = []
    for index in range(start, end):
        scores.append(float(np.exp(logprobs[index, labels[index]])))
    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))
