CC       = cc
CFLAGS_NAIVE = -O0 -g -Wall -Wextra
CFLAGS_OPT   = -O3 -march=native -Wall -Wextra
LDLIBS   = -lm

.PHONY: all clean data bench_tiktoken bench_hf bench_python
.PHONY: bench_naive bench_opt bench_compare

all: bench_naive

# -------------------------------------------------------------------
#  Naive BPE tokenizer (intentionally slow, -O0, array shifting)
# -------------------------------------------------------------------
bench_naive: naive_bpe data/wikitext_test.txt
	@echo "============================================"
	@echo "  Naive BPE Tokenizer (array + O(N²) scan)"
	@echo "  (using 50KB sample — full 50MB would take hours)"
	@echo "============================================"
	head -c 50000 data/wikitext_test.txt > /tmp/naive_sample.txt
	./naive_bpe /tmp/naive_sample.txt 500 1

naive_bpe: naive_bpe.c
	$(CC) $(CFLAGS_NAIVE) -o $@ $< $(LDLIBS)

# -------------------------------------------------------------------
#  Optimized BPE tokenizer (arena list + min-heap, -O3)
# -------------------------------------------------------------------
bench_opt: opt_bpe data/wikitext_test.txt
	@echo "============================================"
	@echo "  Optimized BPE Tokenizer (arena+heap)"
	@echo "  (500 merges, 50KB sample — 5000 on full file would be slow)"
	@echo "============================================"
	head -c 50000 data/wikitext_test.txt > /tmp/opt_sample.txt
	./opt_bpe /tmp/opt_sample.txt 500 1

opt_bpe: opt_bpe.c
	$(CC) $(CFLAGS_OPT) -o $@ $< $(LDLIBS)

# -------------------------------------------------------------------
#  Side-by-side comparison
# -------------------------------------------------------------------
bench_compare: data/wikitext_test.txt naive_bpe opt_bpe
	@echo "============================================"
	@echo "  NAIVE BPE (50KB sample, 500 merges)"
	@echo "============================================"
	head -c 50000 data/wikitext_test.txt > /tmp/naive_sample.txt
	./naive_bpe /tmp/naive_sample.txt 500 1
	@echo ""
	@echo "============================================"
	@echo "  OPTIMIZED BPE (50KB, 500 merges)"
	@echo "============================================"
	head -c 50000 data/wikitext_test.txt > /tmp/opt_sample.txt
	./opt_bpe /tmp/opt_sample.txt 500 1
	@echo ""
	@echo "============================================"
	@echo "  OPTIMIZED BPE (full 50MB, 5000 merges)"
	@echo "============================================"
	./opt_bpe data/wikitext_test.txt 5000 1
	@echo ""
	@echo "============================================"
	@echo "  PYTHON BASELINES"
	@echo "============================================"
	python3 bench_python.py data/wikitext_test.txt --backend all --repeat 3

# -------------------------------------------------------------------
#  Dataset
# -------------------------------------------------------------------
data:
	python3 download_data.py

data/wikitext_test.txt: data
	@ls -lh $@

# -------------------------------------------------------------------
#  Python benchmarks
# -------------------------------------------------------------------
bench_tiktoken: data/wikitext_test.txt
	@echo "============================================"
	@echo "  Python tiktoken (GPT-4 / cl100k_base)"
	@echo "============================================"
	python3 bench_python.py $< --backend tiktoken --repeat 3

bench_hf: data/wikitext_test.txt
	@echo "============================================"
	@echo "  Python HF tokenizers (GPT-2 BPE)"
	@echo "============================================"
	python3 bench_python.py $< --backend hf --repeat 3

bench_python: data/wikitext_test.txt
	@echo "============================================"
	@echo "  Python Tokenizer Benchmarks (all backends)"
	@echo "============================================"
	python3 bench_python.py $< --backend all --repeat 3

# -------------------------------------------------------------------
#  Full benchmark suite
# -------------------------------------------------------------------
bench: bench_compare

# -------------------------------------------------------------------
#  Cleanup
# -------------------------------------------------------------------
clean:
	rm -f naive_bpe opt_bpe
	rm -rf data/

clobber: clean
	rm -rf __pycache__ *.pyc *.dSYM/