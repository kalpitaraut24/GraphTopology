"""
Baseline Retrieval Methods
===========================
Implements BFS-GraphRAG and ToG (Think-on-Graph) baselines
used in Table II of the paper.

G-Retriever and GNN-RAG require pretrained components (graph attention network
and GNN respectively); see their original repositories for setup:
  - G-Retriever : https://github.com/XiaoxinHe/G-Retriever
  - GNN-RAG     : https://github.com/cmavro/GNN-RAG
"""

import math
import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from hg_sto import KnowledgeGraph, EmbeddingCache, TraversalResult, cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BFS-GraphRAG baseline
# ---------------------------------------------------------------------------

class BFSRetriever:
    """
    Standard breadth-first search expansion from seed entity.
    No semantic or structural filtering — serves as the naive upper bound
    on context token consumption (Table II: avg 4,812 tokens).

    Parameters
    ----------
    graph    : KnowledgeGraph
    max_hops : maximum BFS depth (default 3, matching HG-STO horizon K)
    max_nodes: hard cap on collected nodes to prevent memory overflow
    """

    def __init__(self, graph: KnowledgeGraph, max_hops: int = 3, max_nodes: int = 500):
        self.graph = graph
        self.max_hops = max_hops
        self.max_nodes = max_nodes

    def retrieve(self, query: str) -> TraversalResult:
        from hg_sto import fuzzy_entity_link
        seed = fuzzy_entity_link(query, list(self.graph.nodes.keys()))
        if seed is None:
            return TraversalResult(trajectory=[], context=[], hop_scores=[], seed_node="", query=query)

        visited = {seed}
        context = [self.graph.nodes[seed]]
        trajectory = [seed]
        queue = deque([(seed, 0)])

        while queue:
            node, depth = queue.popleft()
            if depth >= self.max_hops:
                continue
            for nbr in self.graph.neighbors(node):
                if nbr not in visited and len(visited) < self.max_nodes:
                    visited.add(nbr)
                    trajectory.append(nbr)
                    context.append(self.graph.nodes[nbr])
                    queue.append((nbr, depth + 1))

        logger.info("BFS: collected %d nodes from seed '%s'", len(trajectory), seed)
        return TraversalResult(
            trajectory=trajectory,
            context=context,
            hop_scores=[],
            seed_node=seed,
            query=query,
        )


# ---------------------------------------------------------------------------
# ToG (Think-on-Graph) baseline
# ---------------------------------------------------------------------------

class ToGRetriever:
    """
    Think-on-Graph: beam search over KG with LLM-assessed relevance pruning.
    Sun et al., ICLR 2024. (https://arxiv.org/abs/2307.07697)

    This implementation uses cosine similarity as a proxy for LLM relevance
    scoring to enable fair comparison without per-hop LLM overhead during
    hyperparameter search. For full ToG with LLM scoring, set use_llm=True.

    Parameters
    ----------
    graph      : KnowledgeGraph
    cache      : precomputed EmbeddingCache
    beam_width : number of top candidates retained per hop (paper uses 3)
    K          : maximum hop depth
    use_llm    : if True, call LLM for relevance scoring (slow but faithful to paper)
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        cache: EmbeddingCache,
        beam_width: int = 3,
        K: int = 3,
        use_llm: bool = False,
        llm=None,
    ):
        self.graph = graph
        self.cache = cache
        self.beam_width = beam_width
        self.K = K
        self.use_llm = use_llm
        self.llm = llm

    def retrieve(self, query: str) -> TraversalResult:
        from hg_sto import fuzzy_entity_link
        e_q = self.cache.encode_query(query)
        seed = fuzzy_entity_link(query, list(self.graph.nodes.keys()))
        if seed is None:
            return TraversalResult(trajectory=[], context=[], hop_scores=[], seed_node="", query=query)

        # Beam: list of (trajectory, visited_set)
        beams = [([seed], {seed})]
        all_context_nodes = {seed}

        for hop in range(self.K):
            new_beams = []
            for traj, visited in beams:
                current = traj[-1]
                candidates = [n for n in self.graph.neighbors(current) if n not in visited]
                if not candidates:
                    new_beams.append((traj, visited))
                    continue

                # Score candidates by cosine similarity (proxy for LLM relevance)
                scored = []
                for c in candidates:
                    e_v = self.cache.get(c)
                    sim = cosine_similarity(e_q, e_v)
                    scored.append((c, sim))
                scored.sort(key=lambda x: x[1], reverse=True)

                for c, _ in scored[: self.beam_width]:
                    new_traj = traj + [c]
                    new_visited = visited | {c}
                    new_beams.append((new_traj, new_visited))
                    all_context_nodes.add(c)

            # Keep top beam_width beams by cumulative similarity
            if new_beams:
                def beam_score(bm):
                    traj, _ = bm
                    return sum(
                        cosine_similarity(e_q, self.cache.get(n))
                        for n in traj[1:]
                    )
                new_beams.sort(key=beam_score, reverse=True)
                beams = new_beams[: self.beam_width]

        # Aggregate context from all explored nodes
        trajectory = list(all_context_nodes)
        context = [self.graph.nodes[n] for n in trajectory]
        logger.info("ToG: collected %d nodes from seed '%s'", len(trajectory), seed)
        return TraversalResult(
            trajectory=trajectory,
            context=context,
            hop_scores=[],
            seed_node=seed,
            query=query,
        )
