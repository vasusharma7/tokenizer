CC       = cc
CFLAGS   = -O3 -Wall -Wextra -march=native
LDLIBS   = -lm

.PHONY: clean data bench

all: bench_tokenizer

# Download dataset
data:
	python3 download_data.py

# Full benchmark: data + C + Python
bench: data bench_tokenizer
	@echo ""
	@echo "============================================"
	@echo "  Python Tokenizer Benchmarks"
	@echo "============================================"
	python3 bench_python.py data/wikitext_test.txt --backend all --repeat 3

clean:
	rm -rf data/