# benchmark/Makefile — Python baselines + data only
# The root Makefile at ../Makefile orchestrates the C tokenizers.

.PHONY: all data clean bench_tiktoken bench_hf bench_python

all: data

# -------------------------------------------------------------------
#  Dataset
# -------------------------------------------------------------------
data:
	python3 download_data.py

data/wikitext_test.txt: data
	@ls -lh $@

# -------------------------------------------------------------------
#  Python baselines
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
#  Cleanup
# -------------------------------------------------------------------
clean:
	rm -rf data/