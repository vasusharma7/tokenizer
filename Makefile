CC          = cc
CFLAGS_NAIVE = -O0 -g -Wall -Wextra
CFLAGS_OPT   = -O3 -march=native -Wall -Wextra
LDLIBS      = -lm

DATA         = benchmark/data/wikitext_test.txt

.PHONY: all clean bench_naive bench_opt bench_python bench_compare data

all: bench_naive

# ===================================================================
#  C Tokenizers (root level)
# ===================================================================

naive_bpe: naive_bpe.c
	$(CC) $(CFLAGS_NAIVE) -o $@ $< $(LDLIBS)

opt_bpe: opt_bpe.c
	$(CC) $(CFLAGS_OPT) -o $@ $< $(LDLIBS)

# ===================================================================
#  Data
# ===================================================================
data: $(DATA)

$(DATA):
	cd benchmark && python3 download_data.py

# ===================================================================
#  Benchmarks
# ===================================================================

bench_naive: naive_bpe $(DATA)
	@echo "============================================"
	@echo "  Naive BPE (array + O(N²) scan)"
	@echo "  (50KB sample, 500 merges — full file would take hours)"
	@echo "============================================"
	head -c 50000 $(DATA) > /tmp/naive_sample.txt
	./naive_bpe /tmp/naive_sample.txt 500 1

bench_opt: opt_bpe $(DATA)
	@echo "============================================"
	@echo "  Optimized BPE (arena + heap + touched-list)"
	@echo "============================================"
	head -c 50000 $(DATA) > /tmp/opt_sample.txt
	./opt_bpe /tmp/opt_sample.txt 500 1

bench_opt_stress: opt_bpe $(DATA)
	@echo "============================================"
	@echo "  Optimized BPE — stress test (500KB, 5000 merges)"
	@echo "============================================"
	head -c 500000 $(DATA) > /tmp/opt_stress.txt
	./opt_bpe /tmp/opt_stress.txt 5000 1

# Python baselines (delegates to benchmark/Makefile)
bench_python:
	cd benchmark && $(MAKE) bench_python

bench_tiktoken:
	cd benchmark && $(MAKE) bench_tiktoken

bench_hf:
	cd benchmark && $(MAKE) bench_hf

# ===================================================================
#  Full comparison
# ===================================================================
bench_compare: bench_naive bench_opt bench_python
	@echo ""
	@echo "============================================"
	@echo "  Full comparison complete!"
	@echo "============================================"

# ===================================================================
#  Cleanup
# ===================================================================
clean:
	rm -f naive_bpe opt_bpe
	rm -rf benchmark/data/ __pycache__
	rm -rf naive_bpe.dSYM/ opt_bpe.dSYM/

clobber: clean
	find . -name '*.pyc' -delete
	rm -rf *.dSYM/