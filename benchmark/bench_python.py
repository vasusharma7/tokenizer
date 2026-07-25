#!/usr/bin/env python3
"""
Benchmark Python tokenizers: tiktoken (OpenAI) and HuggingFace tokenizers.

Measures throughput in tokens/sec and MB/sec.
"""

import argparse
import os
import time
import sys

# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _bench_tiktoken(text: bytes, repeat: int = 3):
    """Benchmark tiktoken (GPT-4 / cl100k_base)."""
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    content = text.decode("utf-8", errors="replace")
    raw_len = len(text)

    # warmup
    _ = enc.encode(content[:1000])

    times = []
    token_counts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        ids = enc.encode(content)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        token_counts.append(len(ids))

    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    return {
        "backend": "tiktoken (cl100k_base / GPT-4)",
        "raw_bytes": raw_len,
        "num_tokens": int(avg_tokens),
        "elapsed_s": avg_time,
        "tokens_per_sec": avg_tokens / avg_time,
        "mb_per_sec": (raw_len / 1e6) / avg_time,
    }


def _bench_hf_tokenizers(text: bytes, repeat: int = 3):
    """Benchmark HuggingFace tokenizers using a pre-trained GPT-2 tokenizer."""
    from tokenizers import Tokenizer
    content = text.decode("utf-8", errors="replace")

    # Load pre-trained GPT-2 tokenizer (ByteLevel BPE, ~50k vocab)
    tokenizer = Tokenizer.from_pretrained("gpt2")

    raw_len = len(text)
    times = []
    token_counts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        encoded = tokenizer.encode(content)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        token_counts.append(len(encoded.ids))

    avg_time = sum(times) / len(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    return {
        "backend": "HF tokenizers (GPT-2 ByteLevel BPE, vocab=50257)",
        "raw_bytes": raw_len,
        "num_tokens": int(avg_tokens),
        "elapsed_s": avg_time,
        "tokens_per_sec": avg_tokens / avg_time,
        "mb_per_sec": (raw_len / 1e6) / avg_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

BACKENDS = {
    "tiktoken": _bench_tiktoken,
    "hf": _bench_hf_tokenizers,
    "all": None,   # special: run all
}


def main():
    parser = argparse.ArgumentParser(description="Benchmark Python tokenizers")
    parser.add_argument("input", help="Path to input text file")
    parser.add_argument(
        "--backend", choices=list(BACKENDS.keys()), default="all",
        help="Tokenizer backend to benchmark",
    )
    parser.add_argument("--repeat", type=int, default=3, help="Repeat count")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "rb") as f:
        text = f.read()

    print(f"Input: {args.input} ({len(text) / 1e6:.1f} MB)")
    print(f"{'='*80}")

    backends = list(BACKENDS.keys() - {"all"}) if args.backend == "all" else [args.backend]

    for backend in backends:
        fn = BACKENDS[backend]
        try:
            result = fn(text, repeat=args.repeat)
            _print_result(result)
        except ImportError as e:
            print(f"[SKIP] {backend} — {e}")
        except Exception as e:
            print(f"[ERROR] {backend} — {e}")
        print()


def _print_result(r: dict):
    print(f"  Backend:        {r['backend']}")
    print(f"  Raw bytes:      {r['raw_bytes']:,}")
    print(f"  Num tokens:     {r['num_tokens']:,}")
    print(f"  Elapsed:        {r['elapsed_s']:.4f} s")
    print(f"  Throughput:     {r['tokens_per_sec']:>12,.0f} tokens/sec")
    print(f"                  {r['mb_per_sec']:>12.2f} MB/sec")


if __name__ == "__main__":
    main()