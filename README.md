# HG-STO: Heuristic-Guided Subgraph Trajectory Optimization

Official implementation for the paper:

> **Topology-Aware Path Pruning in GraphRAG for Accelerated Multi-Hop Relation Extraction**  
> [Author Name], [Co-Author Name]  
> Department of Computer Science, [Institution]

---

## Overview

HG-STO is a **training-free**, **deterministic** framework that resolves the combinatorial path explosion problem in multi-hop GraphRAG systems. It embeds a logarithmic node-degree penalization gate directly into the traversal scoring heuristic, suppressing hub-node detours at runtime without model retraining or reinforcement learning.

**Scoring function:**

```
S(v) = α · Sim(e_q, e_v)  −  β · ln(Deg(v) + 1)
```

Optimal hyperparameters (Table III): `α = 1.0`, `β = 0.35`, `K = 3`

### Key Results (Table II)

| Method | HotpotQA F1 | MetaQA-3hop Acc. | Avg Tokens | Latency (s) | Token Reduction |
|--------|-------------|------------------|------------|-------------|-----------------|
| BFS-GraphRAG | 61.4 | 73.2 | 4,812 | 3.84 | — |
| ToG | 67.8 | 79.6 | 3,940 | 3.12 | 18.1% |
| G-Retriever | 69.1 | 81.3 | 3,620 | 2.93 | 24.8% |
| GNN-RAG | 70.2 | 82.7 | 3,410 | 2.74 | 29.1% |
| **HG-STO (Ours)** | **71.6** | **84.1** | **2,780** | **2.21** | **42.2%** |

LLM backbone: Llama-3-8B-Instruct (frozen, identical across all methods).

---

## Repository Structure

```
hg-sto/
├── src/
│   ├── hg_sto.py          # Core traversal engine (Algorithm 1)
│   ├── baselines.py        # BFS-GraphRAG and ToG baselines
│   ├── metrics.py          # F1, exact-match, bootstrap CI
│   ├── llm_interface.py    # Llama-3 wrapper
│   └── utils.py            # KG loader, dataset loader, prompt builder
├── scripts/
│   ├── evaluate.py         # Main benchmark evaluation script
│   ├── hyperparam_search.py # Grid search reproducing Table III
│   └── clinical_case_study.py  # Reproduces Table I step-by-step
├── configs/
│   ├── hg_sto_optimal.json  # Optimal hyperparameters
│   └── hparam_grid.json     # Full grid search config and results
├── data/                    # Place datasets here (see Data Setup below)
├── results/                 # Evaluation outputs written here
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/[author]/hg-sto.git
cd hg-sto
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

For Llama-3 access, set your HuggingFace token:
```bash
export HF_TOKEN=your_token_here
huggingface-cli login
```

---

## Data Setup

### HotpotQA (distractor dev set — 7,405 examples)
```bash
mkdir -p data
wget http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json \
     -O data/hotpot_dev_distractor_v1.json
```

### MetaQA-3hop (test set — 14,274 examples)
```bash
# Clone the MetaQA dataset repository
git clone https://github.com/yuyuz/MetaQA data/metaqa_raw
cp data/metaqa_raw/metaqa/3-hop/qa_test.txt data/metaqa_3hop_test.txt
# Convert to JSON format expected by the loader:
python scripts/convert_metaqa.py --input data/metaqa_3hop_test.txt \
                                  --output data/metaqa_3hop_test.json
```

### Knowledge Graphs
Pre-built KG files (JSON format) for HotpotQA and MetaQA are provided via the
project's HuggingFace dataset card: `[author]/hg-sto-kg-files`.

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('[author]/hg-sto-kg-files', 'hotpotqa_kg.json', local_dir='data/')
hf_hub_download('[author]/hg-sto-kg-files', 'metaqa_kg.json', local_dir='data/')
"
```

---

## Reproducing Results

### Table I — Clinical Case Study (no GPU required)
```bash
python scripts/clinical_case_study.py
```

### Table II — Benchmark Comparison

**HotpotQA:**
```bash
python scripts/evaluate.py \
    --dataset hotpotqa \
    --data_path data/hotpot_dev_distractor_v1.json \
    --kg_path data/hotpotqa_kg.json \
    --method hg_sto \
    --alpha 1.0 --beta 0.35 --K 3 \
    --output_dir results/
```

**MetaQA-3hop:**
```bash
python scripts/evaluate.py \
    --dataset metaqa \
    --data_path data/metaqa_3hop_test.json \
    --kg_path data/metaqa_kg.json \
    --method hg_sto \
    --alpha 1.0 --beta 0.35 --K 3 \
    --output_dir results/
```

**BFS baseline:**
```bash
python scripts/evaluate.py --method bfs --dataset hotpotqa \
    --data_path data/hotpot_dev_distractor_v1.json \
    --kg_path data/hotpotqa_kg.json --output_dir results/
```

### Table III — Hyperparameter Grid Search
```bash
python scripts/hyperparam_search.py \
    --data_path data/hotpot_dev_distractor_v1.json \
    --kg_path data/hotpotqa_kg.json \
    --output_dir results/hparam_search/
```

---

## Hardware

All experiments were run on a single **NVIDIA A100 80GB GPU**.  
Approximate runtime per benchmark evaluation pass: **~4 hours**.  
The clinical case study (`clinical_case_study.py`) runs on CPU in under 1 minute.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{author2025hgsto,
  title     = {Topology-Aware Path Pruning in {GraphRAG} for Accelerated Multi-Hop Relation Extraction},
  author    = {[Author Name] and [Co-Author Name]},
  year      = {2025},
  note      = {arXiv preprint}
}
```

---

## License

MIT License. See `LICENSE` for details.
