# BPE Tokenizer Benchmark Suite

A from-scratch C implementation of Byte-Pair Encoding (BPE) tokenization,
built to study performance bottlenecks and incremental optimization.

## Structure

```
tokenizer/
├── Makefile              # Root — builds and runs everything
├── naive_bpe.c           # Naive BPE (array + O(N) scan + linked-list merge table)
├── opt_bpe.c             # Optimized BPE (arena list + min-heap + touched-list training)
├── benchmark/
│   ├── Makefile          # Python baseline benchmarks + dataset download
│   ├── bench_python.py   # tiktoken + HuggingFace tokenizers benchmarks
│   └── download_data.py  # Downloads Wikitext-2, pads to ~50MB
└── .gitignore
```

## Make Targets

| Command | What it does |
|---|---|
| `make bench_naive` | Naive BPE on 50KB sample, 500 merges |
| `make bench_opt` | Optimized BPE on 50KB sample, 500 merges |
| `make bench_opt_stress` | Optimized BPE on 500KB sample, 5000 merges |
| `make bench_tiktoken` | Python tiktoken (GPT-4) on full 50MB |
| `make bench_hf` | Python HuggingFace tokenizers on full 50MB |
| `make bench_python` | All Python baselines |
| `make bench_compare` | Naive + opt + Python side-by-side |
| `make clean` | Remove binaries and dataset |

## Build Requirements

- C compiler (clang or gcc)
- Python 3 with tiktoken, tokenizers, datasets packages
