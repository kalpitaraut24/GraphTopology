"""
Benchmark Evaluation Script
============================
Evaluates HG-STO (and baselines) on HotpotQA and MetaQA-3hop.

Usage
-----
# HotpotQA (distractor dev set, 7,405 examples):
python scripts/evaluate.py \
    --dataset hotpotqa \
    --data_path data/hotpot_dev_distractor_v1.json \
    --kg_path data/hotpotqa_kg.json \
    --method hg_sto \
    --alpha 1.0 --beta 0.35 --K 3 \
    --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir results/

# MetaQA-3hop (test set, 14,274 examples):
python scripts/evaluate.py \
    --dataset metaqa \
    --data_path data/metaqa_3hop_test.json \
    --kg_path data/metaqa_kg.json \
    --method hg_sto \
    --alpha 1.0 --beta 0.35 --K 3 \
    --llm_model meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir results/

Methods: hg_sto | bfs | tog | g_retriever | gnn_rag
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from hg_sto import HGSTO, KnowledgeGraph, EmbeddingCache
from baselines import BFSRetriever, ToGRetriever
from metrics import compute_f1, compute_exact_match, bootstrap_confidence_interval
from llm_interface import LLMInterface
from utils import load_kg, load_dataset, build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="HG-STO benchmark evaluation")

    # Data
    parser.add_argument("--dataset", choices=["hotpotqa", "metaqa"], required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--kg_path", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit evaluation to N examples (None = full set)")

    # Method
    parser.add_argument("--method", choices=["hg_sto", "bfs", "tog", "g_retriever", "gnn_rag"],
                        default="hg_sto")

    # HG-STO hyperparameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Semantic alignment weight (default: 1.0)")
    parser.add_argument("--beta", type=float, default=0.35,
                        help="Structural penalization weight (default: 0.35)")
    parser.add_argument("--K", type=int, default=3,
                        help="Maximum traversal horizon (default: 3)")
    parser.add_argument("--tau", type=float, default=0.85,
                        help="Entity linking similarity threshold (default: 0.85)")

    # Encoder
    parser.add_argument("--encoder", type=str, default="all-MiniLM-L6-v2",
                        help="SentenceTransformer model name")

    # LLM
    parser.add_argument("--llm_model", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=128)

    # Output
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--latency_sample_size", type=int, default=500,
                        help="Number of examples used for latency measurement")

    # Bootstrap significance
    parser.add_argument("--bootstrap_n", type=int, default=1000,
                        help="Bootstrap resampling iterations for significance testing")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Retriever factory
# ---------------------------------------------------------------------------

def build_retriever(args, graph: KnowledgeGraph, cache: EmbeddingCache):
    if args.method == "hg_sto":
        return HGSTO(graph, cache, alpha=args.alpha, beta=args.beta, K=args.K, tau=args.tau)
    elif args.method == "bfs":
        return BFSRetriever(graph, max_hops=args.K)
    elif args.method == "tog":
        return ToGRetriever(graph, cache, beam_width=3, K=args.K)
    else:
        raise NotImplementedError(
            f"Method '{args.method}' requires additional setup. "
            f"See baselines.py for G-Retriever and GNN-RAG."
        )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Approximate token count (whitespace split; ~0.75 words per token)."""
    return max(1, int(len(text.split()) / 0.75))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load KG and dataset
    logger.info("Loading knowledge graph from %s", args.kg_path)
    graph = load_kg(args.kg_path)
    logger.info("Graph: %d nodes, %d edge entries", len(graph.nodes), sum(len(v) for v in graph.edges.values()))

    logger.info("Loading dataset from %s", args.data_path)
    examples = load_dataset(args.data_path, args.dataset)
    if args.max_samples:
        examples = examples[: args.max_samples]
    logger.info("Evaluating %d examples", len(examples))

    # Precompute embeddings
    cache = EmbeddingCache(model_name=args.encoder)
    cache.precompute(graph)

    # Build retriever and LLM
    retriever = build_retriever(args, graph, cache)
    llm = LLMInterface(model_name=args.llm_model, max_new_tokens=args.max_new_tokens)

    # Evaluation loop
    scores = []
    token_counts = []
    latencies = []
    predictions = []

    for i, ex in enumerate(tqdm(examples, desc=f"Evaluating [{args.method}]")):
        query = ex["question"]
        gold = ex["answer"]

        t0 = time.perf_counter()

        # Retrieval
        try:
            result = retriever.traverse(query) if hasattr(retriever, "traverse") else retriever.retrieve(query)
            context_text = "\n".join(result.context) if hasattr(result, "context") else result
        except Exception as e:
            logger.warning("Retrieval failed for example %d: %s", i, e)
            context_text = ""

        # LLM generation
        prompt = build_prompt(query, context_text, dataset=args.dataset)
        prediction = llm.generate(prompt)

        latency = time.perf_counter() - t0

        # Metrics
        if args.dataset == "hotpotqa":
            score = compute_f1(prediction, gold)
        else:
            score = compute_exact_match(prediction, gold)

        token_count = count_tokens(context_text)
        scores.append(score)
        token_counts.append(token_count)
        predictions.append({"id": ex.get("id", i), "prediction": prediction, "gold": gold, "score": score})

        if i < args.latency_sample_size:
            latencies.append(latency)

    # Aggregate metrics
    mean_score = np.mean(scores)
    mean_tokens = np.mean(token_counts)
    mean_latency = np.mean(latencies) if latencies else 0.0

    metric_name = "F1" if args.dataset == "hotpotqa" else "Exact Match Acc."
    logger.info("=" * 60)
    logger.info("Method:        %s", args.method)
    logger.info("Dataset:       %s", args.dataset)
    logger.info("%s:  %.4f", metric_name, mean_score)
    logger.info("Avg Tokens:    %.1f", mean_tokens)
    logger.info("Avg Latency:   %.4f s  (over %d samples)", mean_latency, len(latencies))

    # Bootstrap confidence interval
    ci_low, ci_high = bootstrap_confidence_interval(scores, n=args.bootstrap_n)
    logger.info("95%% CI (%s): [%.4f, %.4f]", metric_name, ci_low, ci_high)
    logger.info("=" * 60)

    # Save results
    run_name = f"{args.dataset}_{args.method}_a{args.alpha}_b{args.beta}_K{args.K}"
    results = {
        "method": args.method,
        "dataset": args.dataset,
        "alpha": args.alpha,
        "beta": args.beta,
        "K": args.K,
        "tau": args.tau,
        "encoder": args.encoder,
        "llm_model": args.llm_model,
        "n_examples": len(examples),
        metric_name: round(float(mean_score), 4),
        "avg_tokens": round(float(mean_tokens), 1),
        "avg_latency_s": round(float(mean_latency), 4),
        "ci_95_low": round(float(ci_low), 4),
        "ci_95_high": round(float(ci_high), 4),
    }
    out_path = output_dir / f"{run_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", out_path)

    preds_path = output_dir / f"{run_name}_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(predictions, f, indent=2)
    logger.info("Predictions saved to %s", preds_path)

    return results


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
