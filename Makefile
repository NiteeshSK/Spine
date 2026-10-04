# SPINE -- one command per stage of the pipeline.
PY ?= python
export PYTHONPATH := src

BASE ?= Qwen/Qwen3-8B
SFT_CFG ?= configs/sft_qlora.yaml
DPO_CFG ?= configs/dpo.yaml
BACKEND ?= hf

.PHONY: help setup setup-cpu data test smoke sft eval-base eval-sft dpo-pairs dpo eval-dpo eval-ood figures table demo all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "};{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:          ## install the full GPU training stack
	pip install -r requirements.txt
setup-cpu:      ## install only what data/eval/figures need (no CUDA)
	pip install -r requirements-cpu.txt

data:           ## generate the corpus and run the data audit gate
	$(PY) -m spine.data.synth --out data/raw
	$(PY) -m spine.data.build_sft --raw data/raw --out data/sft
	cp data/sft/pool.jsonl data/sft/dpo_pool.jsonl

test:           ## run the unit tests (schema, metrics, generator, audit)
	$(PY) -m pytest

smoke:          ## 3-minute CPU end-to-end check of the training code path
	$(PY) -m spine.train.sft --config configs/sft_smoke.yaml

# ---------------------------------------------------------------- phase 1 ---
sft:            ## QLoRA supervised fine-tune
	$(PY) -m spine.train.sft --config $(SFT_CFG)

eval-base:      ## baseline: base weights + full schema prompt + 4 shots
	$(PY) -m spine.eval.run_eval --split test --backend $(BACKEND) \
		--model $(BASE) --prompt baseline --tag base_prompted

eval-sft:       ## evaluate the SFT adapter
	$(PY) -m spine.eval.run_eval --split test --backend $(BACKEND) \
		--model $(BASE) --adapter runs/sft/adapter --tag sft

# ---------------------------------------------------------------- phase 2 ---
dpo-pairs:      ## sample k candidates on-policy and mine preference pairs
	$(PY) -m spine.data.build_dpo --model $(BASE) --adapter runs/sft/adapter \
		--backend $(BACKEND) --pool data/sft/dpo_pool.jsonl --k 6 --temperature 0.9

dpo:            ## preference-tune on the mined pairs
	$(PY) -m spine.train.dpo --config $(DPO_CFG)

eval-dpo:       ## evaluate the DPO adapter
	$(PY) -m spine.eval.run_eval --split test --backend $(BACKEND) \
		--model $(BASE) --adapter runs/dpo/adapter --tag dpo

eval-ood:       ## same three models on the held-out surface styles
	$(PY) -m spine.eval.run_eval --split ood --model $(BASE) --prompt baseline --tag base_prompted
	$(PY) -m spine.eval.run_eval --split ood --model $(BASE) --adapter runs/sft/adapter --tag sft
	$(PY) -m spine.eval.run_eval --split ood --model $(BASE) --adapter runs/dpo/adapter --tag dpo

# ------------------------------------------------------------------ report ---
figures:        ## regenerate every figure from run artifacts
	$(PY) -m spine.viz.plots --out assets
table:          ## regenerate the README results table
	$(PY) -m spine.eval.compare --split test > docs/_table_test.md
	$(PY) -m spine.eval.compare --split ood  > docs/_table_ood.md
	@cat docs/_table_test.md

demo:           ## end-to-end dry run with simulated outputs (no GPU, no model)
	$(PY) scripts/pipeline_demo.py

all: data test sft eval-base eval-sft dpo-pairs dpo eval-dpo figures table

clean:
	rm -rf runs reports assets/*.png data/raw data/sft data/dpo
