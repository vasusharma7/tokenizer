CC       = cc
CFLAGS   = -O0 -g -Wall -Wextra
LDLIBS   = -lm

.PHONY: all clean data bench_tiktoken bench_hf bench_python bench_naive

all: bench_naive

# -------------------------------------------------------------------
#  Naive BPE tokenizer (intentionally slow, -O0)
# -------------------------------------------------------------------
bench_naive: naive_bpe
	@echo "============================================"
	@echo "  Naive BPE Tokenizer (no optimizations)"
	@echo "============================================"
	./naive_bpe data/wikitext_test.txt 5000 1

naive_bpe: naive_bpe.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

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
bench: data/wikitext_test.txt bench_naive bench_python

# -------------------------------------------------------------------
#  Cleanup
# -------------------------------------------------------------------
clean:
	rm -f naive_bpe
	rm -rf data/

clobber: clean
	rm -rf __pycache__ *.pyc