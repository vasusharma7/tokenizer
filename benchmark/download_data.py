#!/usr/bin/env python3
"""
Download Wikitext-2 raw dataset for benchmarking.
Tries HuggingFace datasets first, falls back to synthetic text.
"""

import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "wikitext_test.txt")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(OUTPUT_FILE):
        size = os.path.getsize(OUTPUT_FILE)
        print(f"Dataset already exists at {OUTPUT_FILE} ({size / 1e6:.1f} MB)")
        return

    # Try HuggingFace datasets library
    try:
        from datasets import load_dataset
        print("Downloading Wikitext-2 via HuggingFace datasets ...")
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
        with open(OUTPUT_FILE, "w") as f:
            for split in ["train", "test", "validation"]:
                for row in ds[split]:
                    f.write(row["text"] + "\n")
        size = os.path.getsize(OUTPUT_FILE)
        print(f"Dataset ready: {OUTPUT_FILE} ({size / 1e6:.1f} MB)")
        if size < 50e6:
            # Append synthetic text to reach ~50MB for meaningful benchmarks
            _append_synthetic(OUTPUT_FILE, target_mb=50)
        return
    except Exception as e:
        print(f"HuggingFace datasets failed: {e}")
        print("Falling back to synthetic text generation...")

    _generate_synthetic(OUTPUT_FILE)
    size = os.path.getsize(OUTPUT_FILE)
    print(f"Dataset ready: {OUTPUT_FILE} ({size / 1e6:.1f} MB)")


def _generate_synthetic(path: str, target_mb: int = 50):
    """Generate synthetic English-like text if download fails."""
    _write_synthetic(path, target_mb)


def _append_synthetic(path: str, target_mb: int = 50):
    """Append synthetic text to reach target_mb total size."""
    current = os.path.getsize(path)
    needed = target_mb * 1024 * 1024 - current
    if needed > 0:
        print(f"Appending {needed / 1e6:.1f}MB of synthetic text to reach ~{target_mb}MB")
        _write_synthetic(path, target_mb, append=True)


def _write_synthetic(path: str, target_mb: int, append: bool = False):
    import random
    random.seed(42)

    words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
        "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
        "take", "people", "into", "year", "your", "good", "some", "could", "them",
        "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
        "think", "also", "back", "after", "use", "two", "how", "our", "work",
        "first", "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "Transformer", "attention", "tokenizer",
        "neural", "network", "deep", "learning", "embedding", "vector",
        "language", "model", "encoder", "decoder", "attention", "layer",
        "gradient", "optimization", "training", "inference", "batch",
    ]
    target_bytes = target_mb * 1024 * 1024
    mode = "a" if append else "w"
    written = 0
    if append:
        written = os.path.getsize(path)

    with open(path, mode) as f:
        while written < target_bytes:
            line_len = random.randint(5, 30)
            line_words = [random.choice(words) for _ in range(line_len)]
            sentence = " ".join(line_words) + "\n"
            f.write(sentence)
            written += len(sentence)

    if append:
        print(f"Appended text to reach {target_mb}MB total.")
    else:
        print(f"Generated {target_mb}MB synthetic text as fallback.")


if __name__ == "__main__":
    main()