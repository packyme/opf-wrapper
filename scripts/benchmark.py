import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm


DEFAULT_TEXT = (
    "My name is Alice Smith and my email is alice@example.com. "
    "Call me at +1 415 555 0134."
)
TEXT_MODES = ("fixed", "short", "medium", "long", "mixed")
MIXED_MODES = ("short", "medium", "long")
PII_SNIPPETS = (
    "Customer John Smith can be reached at john.smith@gmail.com or +1 415 555 0134.",
    "Billing address: 123 Main Street, New York, NY 10001.",
    "Account number 4111-1111-1111-1111 was used for the reservation.",
    "Emergency contact Alice Johnson uses alice.johnson@example.com.",
    "Support token sk-test-51H8aD2exampleSecretValue should be rotated.",
    "The backup phone number is 13800138000.",
)
FILLER_SENTENCES = (
    "The operations team reviewed the document before sending it to the archive.",
    "This paragraph contains ordinary business context without sensitive identifiers.",
    "The service records validation details, processing status, and audit notes.",
    "A downstream system consumes the sanitized payload during nightly jobs.",
    "The text may include repeated sections, punctuation, and mixed formatting.",
    "Performance should be measured across realistic request sizes.",
)


def main() -> None:
    args = parse_args()
    url = build_url(args.base_url, args.endpoint)
    rng = random.Random(args.seed)
    warmup_payloads = build_payloads(args, args.warmup, rng)
    payloads = build_payloads(args, args.requests, rng)
    show_progress = not args.no_progress

    check_health(args.base_url, args.timeout)
    run_warmup(url, warmup_payloads, args.timeout, show_progress)

    started_at = time.perf_counter()
    results = run_requests(url, payloads, args.concurrency, args.timeout, show_progress)
    elapsed = time.perf_counter() - started_at

    print_report(url, args, results, elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the OPF HTTP wrapper.")
    parser.add_argument("--base-url", default=os.getenv("OPF_BENCHMARK_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--endpoint", default=os.getenv("OPF_BENCHMARK_ENDPOINT", "/detect"))
    parser.add_argument("--requests", type=int, default=env_int("OPF_BENCHMARK_REQUESTS", 50))
    parser.add_argument("--warmup", type=int, default=env_int("OPF_BENCHMARK_WARMUP", 5))
    parser.add_argument("--concurrency", type=int, default=env_int("OPF_BENCHMARK_CONCURRENCY", 1))
    parser.add_argument("--threshold", type=float, default=env_float("OPF_BENCHMARK_THRESHOLD", 0.0))
    parser.add_argument("--timeout", type=float, default=env_float("OPF_BENCHMARK_TIMEOUT", 60.0))
    parser.add_argument("--text-mode", choices=TEXT_MODES, default=os.getenv("OPF_BENCHMARK_TEXT_MODE", "mixed"))
    parser.add_argument("--short-chars", type=int, default=env_int("OPF_BENCHMARK_SHORT_CHARS", 256))
    parser.add_argument("--medium-chars", type=int, default=env_int("OPF_BENCHMARK_MEDIUM_CHARS", 4096))
    parser.add_argument("--long-chars", type=int, default=env_int("OPF_BENCHMARK_LONG_CHARS", 32768))
    parser.add_argument("--seed", type=int, default=env_int("OPF_BENCHMARK_SEED", 7))
    parser.add_argument("--no-progress", action="store_true", default=env_bool("OPF_BENCHMARK_NO_PROGRESS", False))
    parser.add_argument("--text", default=os.getenv("OPF_BENCHMARK_TEXT"))
    parser.add_argument("--text-file", default=os.getenv("OPF_BENCHMARK_TEXT_FILE"))
    return parser.parse_args()


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def build_url(base_url: str, endpoint: str) -> str:
    normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    return f"{base_url.rstrip('/')}{normalized_endpoint}"


def build_payloads(
    args: argparse.Namespace,
    total: int,
    rng: random.Random,
) -> list[tuple[str, dict[str, float | str]]]:
    if total <= 0:
        return []

    if args.text_file:
        with open(args.text_file, encoding="utf-8") as file:
            return [("fixed", build_payload(file.read(), args.threshold)) for _ in range(total)]
    if args.text:
        return [("fixed", build_payload(args.text, args.threshold)) for _ in range(total)]
    if args.text_mode == "fixed":
        return [("fixed", build_payload(DEFAULT_TEXT, args.threshold)) for _ in range(total)]

    payloads: list[tuple[str, dict[str, float | str]]] = []
    modes = MIXED_MODES if args.text_mode == "mixed" else (args.text_mode,)
    for _ in range(total):
        mode = rng.choice(modes)
        text = generate_text(mode, args, rng)
        payloads.append((mode, build_payload(text, args.threshold)))
    return payloads


def build_payload(text: str, threshold: float) -> dict[str, float | str]:
    return {
        "text": text,
        "threshold": threshold,
    }


def generate_text(mode: str, args: argparse.Namespace, rng: random.Random) -> str:
    target_chars = {
        "short": args.short_chars,
        "medium": args.medium_chars,
        "long": args.long_chars,
    }[mode]

    parts: list[str] = []
    while len(" ".join(parts)) < target_chars:
        parts.append(rng.choice(FILLER_SENTENCES))
        if rng.random() < 0.35:
            parts.append(rng.choice(PII_SNIPPETS))

    return " ".join(parts)


def check_health(base_url: str, timeout: float) -> None:
    health_url = build_url(base_url, "/health")
    try:
        request_json("GET", health_url, None, timeout)
    except Exception as exc:
        raise SystemExit(f"Health check failed: {health_url}\n{exc}") from exc


def run_warmup(
    url: str,
    payloads: list[tuple[str, dict[str, float | str]]],
    timeout: float,
    show_progress: bool,
) -> None:
    if not payloads:
        return

    for _, payload in tqdm(payloads, desc="Warmup", unit="req", disable=not show_progress):
        request_json("POST", url, payload, timeout)


def run_requests(
    url: str,
    payloads: list[tuple[str, dict[str, float | str]]],
    concurrency: int,
    timeout: float,
    show_progress: bool,
) -> list[tuple[str, int, float]]:
    if not payloads:
        raise SystemExit("--requests must be greater than 0")
    if concurrency <= 0:
        raise SystemExit("--concurrency must be greater than 0")

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(timed_request, url, bucket, payload, timeout) for bucket, payload in payloads]
        completed = as_completed(futures)
        progress = tqdm(completed, total=len(futures), desc="Benchmark", unit="req", disable=not show_progress)
        return [future.result() for future in progress]


def timed_request(url: str, bucket: str, payload: dict[str, float | str], timeout: float) -> tuple[str, int, float]:
    started_at = time.perf_counter()
    request_json("POST", url, payload, timeout)
    text = payload["text"]
    return bucket, len(str(text)), time.perf_counter() - started_at


def request_json(method: str, url: str, payload: dict[str, float | str] | None, timeout: float) -> dict:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    if not body:
        return {}
    return json.loads(body)


def print_report(url: str, args: argparse.Namespace, results: list[tuple[str, int, float]], elapsed: float) -> None:
    latencies_ms = sorted(latency * 1000 for _, _, latency in results)
    text_lengths = [text_length for _, text_length, _ in results]

    print()
    print("Benchmark result")
    print(f"  url:         {url}")
    print(f"  requests:    {args.requests}")
    print(f"  warmup:      {args.warmup}")
    print(f"  concurrency: {args.concurrency}")
    print(f"  text mode:   {get_text_mode_label(args, results)}")
    print(f"  text chars:  {min(text_lengths)} min / {round(mean(text_lengths))} mean / {max(text_lengths)} max")
    print(f"  total:       {elapsed:.3f}s")
    print(f"  rps:         {args.requests / elapsed:.2f}")
    print()
    print_latency("Latency", latencies_ms)
    print_bucket_reports(results)


def print_bucket_reports(results: list[tuple[str, int, float]]) -> None:
    buckets = [bucket for bucket in MIXED_MODES if any(result_bucket == bucket for result_bucket, _, _ in results)]
    if not buckets:
        return

    print()
    print("Latency by text size")
    for bucket in buckets:
        bucket_results = [latency * 1000 for result_bucket, _, latency in results if result_bucket == bucket]
        print_latency(f"  {bucket}", sorted(bucket_results))


def get_text_mode_label(args: argparse.Namespace, results: list[tuple[str, int, float]]) -> str:
    buckets = {bucket for bucket, _, _ in results}
    if buckets == {"fixed"}:
        return "fixed"
    return args.text_mode


def print_latency(title: str, latencies_ms: list[float]) -> None:
    print(title)
    print(f"  count: {len(latencies_ms)}")
    print(f"  min:   {latencies_ms[0]:.2f} ms")
    print(f"  mean:  {mean(latencies_ms):.2f} ms")
    print(f"  p50:   {percentile(latencies_ms, 50):.2f} ms")
    print(f"  p90:   {percentile(latencies_ms, 90):.2f} ms")
    print(f"  p95:   {percentile(latencies_ms, 95):.2f} ms")
    print(f"  p99:   {percentile(latencies_ms, 99):.2f} ms")
    print(f"  max:   {latencies_ms[-1]:.2f} ms")


def percentile(values: list[float], percent: int) -> float:
    if not values:
        raise ValueError("values must not be empty")

    index = round((len(values) - 1) * percent / 100)
    return values[index]


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
