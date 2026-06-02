"""
Utility Functions
=================
KG loading, dataset loading, and prompt construction.
"""

import json
import logging
from pathlib import Path

from hg_sto import KnowledgeGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates (identical across all methods for fair comparison)
# ---------------------------------------------------------------------------

HOTPOTQA_PROMPT = (
    "Answer the following multi-hop question using only the provided context.\n"
    "If the context does not contain the answer, respond with 'unknown'.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

METAQA_PROMPT = (
    "Answer the following knowledge graph question using only the provided context.\n"
    "Give a short, direct answer (entity name only).\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def build_prompt(query: str, context: str, dataset: str = "hotpotqa") -> str:
    """Build the LLM prompt for a given query and retrieved context."""
    template = HOTPOTQA_PROMPT if dataset == "hotpotqa" else METAQA_PROMPT
    return template.format(question=query, context=context[:4000])  # truncate if needed


# ---------------------------------------------------------------------------
# KG loading
# ---------------------------------------------------------------------------

def load_kg(kg_path: str) -> KnowledgeGraph:
    """
    Load a KnowledgeGraph from a JSON file.

    Expected format:
    {
      "nodes": {"node_id": "text description", ...},
      "edges": {"src_id": [["relation", "dst_id"], ...], ...}
    }
    """
    with open(kg_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    graph = KnowledgeGraph()
    for node_id, desc in data["nodes"].items():
        graph.add_node(node_id, desc)
    for src, edge_list in data.get("edges", {}).items():
        for rel, dst in edge_list:
            graph.add_edge(src, rel, dst)

    logger.info("Loaded KG: %d nodes, %d edge entries",
                len(graph.nodes), sum(len(v) for v in graph.edges.values()))
    return graph


def save_kg(graph: KnowledgeGraph, out_path: str):
    """Serialize a KnowledgeGraph to JSON."""
    data = {
        "nodes": dict(graph.nodes),
        "edges": {src: list(edges) for src, edges in graph.edges.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("KG saved to %s", out_path)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(data_path: str, dataset: str) -> list[dict]:
    """
    Load evaluation examples from HotpotQA or MetaQA JSON files.

    HotpotQA format (distractor dev):
        List of {"_id", "question", "answer", "supporting_facts", ...}

    MetaQA-3hop format:
        List of {"question", "answer"} or {"question", "answers"}
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    if dataset == "hotpotqa":
        for item in raw:
            examples.append({
                "id":       item.get("_id", ""),
                "question": item["question"],
                "answer":   item["answer"],
            })
    elif dataset == "metaqa":
        for item in raw:
            answer = item.get("answer") or item.get("answers")
            if isinstance(answer, list):
                answer = answer[0]
            examples.append({
                "id":       item.get("id", ""),
                "question": item["question"],
                "answer":   str(answer),
            })
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    logger.info("Loaded %d examples from %s", len(examples), data_path)
    return examples
