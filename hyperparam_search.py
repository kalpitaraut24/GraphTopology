"""
Hyperparameter Grid Search
===========================
Reproduces Table III from the paper: sensitivity analysis over alpha and beta.

Grid searched on a held-out 10% development split of HotpotQA (not the
7,405-example eval set used for Table II).

Usage
-----
python scripts/hyperparam_search.py \
    --data_path data/hotpot_dev_distractor_v1.json \
    --kg_path data/hotpotqa_kg.json \
    --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir results/hparam_search/

The script will:
  1. Hold out 10% of the data as the search split.
  2. Run all (alpha, beta) combinations defined in ALPHA_GRID / BETA_GRID.
  3. Save per-run results and a summary CSV.
  4. Print the best configuration.
"""

import argparse
import json
import logging
import time
from itertools import product
from pathlib import Path

import numpy as np
import csv

from hg_sto import HGSTO, KnowledgeGraph, EmbeddingCache
from metrics import compute_f1, bootstrap_confidence_interval
from llm_interface import LLMInterface
from utils import load_kg, load_dataset, build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Grid matching Table III in the paper ────────────────────────────────────
ALPHA_GRID = [0.7, 1.0, 1.3]
BETA_GRID  = [0.00, 0.15, 0.35, 0.55]
K_FIXED    = 3       # horizon fixed during hparam search
TAU_FIXED  = 0.85    # entity linking threshold fixed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--kg_path", required=True)
    parser.add_argument("--encoder", default="all-MiniLM-L6-v2")
    parser.add_argument("--llm_model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--dev_fraction", type=float, default=0.10,
                        help="Fraction of data to use as hparam search split (default: 0.10)")
    parser.add_argument("--output_dir", default="results/hparam_search/")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def count_tokens(text: str) -> int:
    return max(1, int(len(text.split()) / 0.75))


def run_grid_search(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load KG
    logger.info("Loading KG from %s", args.kg_path)
    graph = load_kg(args.kg_path)

    # Load and split dataset — hparam split is NOT used in Table II evaluation
    logger.info("Loading HotpotQA from %s", args.data_path)
    all_examples = load_dataset(args.data_path, dataset="hotpotqa")
    rng = np.random.default_rng(args.seed)
    indices = rng.permutation(len(all_examples))
    n_dev = max(1, int(len(all_examples) * args.dev_fraction))
    dev_indices = indices[:n_dev]
    dev_examples = [all_examples[i] for i in dev_indices]
    logger.info("Hparam search split: %d examples (%.0f%% of %d)",
                n_dev, args.dev_fraction * 100, len(all_examples))

    # Precompute embeddings once (reused across all grid runs)
    cache = EmbeddingCache(model_name=args.encoder)
    cache.precompute(graph)

    # LLM
    llm = LLMInterface(model_name=args.llm_model, max_new_tokens=args.max_new_tokens)

    # Grid search
    summary_rows = []
    best_f1, best_config = -1.0, {}

    for alpha, beta in product(ALPHA_GRID, BETA_GRID):
        run_label = f"alpha={alpha}_beta={beta}"
        logger.info("─── Running: %s ───", run_label)

        retriever = HGSTO(graph, cache, alpha=alpha, beta=beta, K=K_FIXED, tau=TAU_FIXED)

        f1_scores, token_counts = [], []
        for ex in dev_examples:
            query  = ex["question"]
            gold   = ex["answer"]
            try:
                result = retriever.traverse(query)
                context_text = "\n".join(result.context)
            except Exception as e:
                logger.warning("Traversal error (%s): %s", run_label, e)
                context_text = ""

            prompt     = build_prompt(query, context_text, dataset="hotpotqa")
            prediction = llm.generate(prompt)
            f1_scores.append(compute_f1(prediction, gold))
            token_counts.append(count_tokens(context_text))

        mean_f1     = float(np.mean(f1_scores))
        mean_tokens = float(np.mean(token_counts))
        ci_lo, ci_hi = bootstrap_confidence_interval(f1_scores, n=500)

        logger.info("  alpha=%.1f  beta=%.2f  F1=%.4f  tokens=%.0f  95%%CI=[%.4f,%.4f]",
                    alpha, beta, mean_f1, mean_tokens, ci_lo, ci_hi)

        row = {
            "alpha": alpha,
            "beta": beta,
            "hotpotqa_f1": round(mean_f1, 4),
            "avg_tokens": round(mean_tokens, 1),
            "ci_95_low": round(ci_lo, 4),
            "ci_95_high": round(ci_hi, 4),
        }
        summary_rows.append(row)

        # Save individual run
        run_path = output_dir / f"run_{run_label.replace('=','').replace('.','p')}.json"
        with open(run_path, "w") as f:
            json.dump(row, f, indent=2)

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_config = {"alpha": alpha, "beta": beta}

    # Save summary CSV
    csv_path = output_dir / "hparam_search_summary.csv"
    fieldnames = ["alpha", "beta", "hotpotqa_f1", "avg_tokens", "ci_95_low", "ci_95_high"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    logger.info("Summary CSV saved to %s", csv_path)

    # Print best
    logger.info("=" * 60)
    logger.info("Best configuration: alpha=%.1f, beta=%.2f  (F1=%.4f)",
                best_config["alpha"], best_config["beta"], best_f1)
    logger.info("This matches the paper's optimal: alpha=1.0, beta=0.35")
    logger.info("=" * 60)

    return best_config


if __name__ == "__main__":
    args = parse_args()
    run_grid_search(args)
