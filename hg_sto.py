"""
HG-STO: Heuristic-Guided Subgraph Trajectory Optimization
==========================================================
Core traversal engine implementing Algorithm 1 from the paper.

Scoring function:
    S(v) = alpha * Sim(e_q, e_v) - beta * ln(Deg(v) + 1)

where Sim is cosine similarity between query and node embeddings,
Deg(v) is the out-degree of candidate node v, and alpha/beta are
hyperparameters (optimal: alpha=1.0, beta=0.35).
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class KnowledgeGraph:
    """
    Directed, text-attributed multi-relational knowledge graph G = (V, E, R).

    Attributes
    ----------
    nodes : dict[str, str]
        Mapping from node_id -> attribute text description D_v.
    edges : dict[str, list[tuple[str, str]]]
        Adjacency list: node_id -> [(relation, neighbor_id), ...].
    """
    nodes: dict = field(default_factory=dict)
    edges: dict = field(default_factory=dict)

    def out_degree(self, node_id: str) -> int:
        return len(self.edges.get(node_id, []))

    def neighbors(self, node_id: str) -> list[str]:
        return [nbr for _, nbr in self.edges.get(node_id, [])]

    def add_node(self, node_id: str, description: str):
        self.nodes[node_id] = description
        if node_id not in self.edges:
            self.edges[node_id] = []

    def add_edge(self, src: str, relation: str, dst: str):
        if src not in self.edges:
            self.edges[src] = []
        self.edges[src].append((relation, dst))


@dataclass
class TraversalResult:
    """Output of a single HG-STO query traversal."""
    trajectory: list[str]          # ordered list of visited node IDs
    context: list[str]             # aggregated text descriptions N(v_t)
    hop_scores: list[dict]         # per-hop scoring details for interpretability
    seed_node: str
    query: str


# ---------------------------------------------------------------------------
# Entity linker
# ---------------------------------------------------------------------------

def fuzzy_entity_link(query: str, node_ids: list[str], threshold: float = 0.85) -> Optional[str]:
    """
    Map a query string to a seed node via fuzzy surface-form matching.
    Uses normalised Levenshtein similarity (1 - edit_distance / max_len).
    Falls back to None if no match exceeds the threshold tau.

    Parameters
    ----------
    query     : raw query string
    node_ids  : list of candidate node identifiers
    threshold : tau in the paper (default 0.85)
    """
    query_lower = query.lower()
    best_node, best_score = None, -1.0

    for node_id in node_ids:
        node_lower = node_id.lower().replace("_", " ")

        # Check for direct substring match first (fast path)
        if node_lower in query_lower:
            return node_id

        # Normalised Levenshtein
        sim = _levenshtein_sim(query_lower, node_lower)
        if sim > best_score:
            best_score = sim
            best_node = node_id

    if best_score >= threshold:
        return best_node

    logger.warning(
        "Entity linking found no match above tau=%.2f (best: %s @ %.3f)",
        threshold, best_node, best_score
    )
    return best_node  # return best effort even below threshold


def _levenshtein_sim(a: str, b: str) -> float:
    """Normalised Levenshtein similarity in [0, 1]."""
    if not a and not b:
        return 1.0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    dist = dp[n]
    return 1.0 - dist / max(m, n)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Precomputes and caches node embeddings for O(1) lookup during traversal.
    Matches the paper's description: embeddings are computed offline,
    reducing per-query overhead to lookup + dot-product.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info("Loading encoder: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self._cache: dict[str, np.ndarray] = {}

    def precompute(self, graph: KnowledgeGraph):
        """Encode all node descriptions and store in cache."""
        node_ids = list(graph.nodes.keys())
        descriptions = [graph.nodes[nid] for nid in node_ids]
        logger.info("Precomputing embeddings for %d nodes ...", len(node_ids))
        embeddings = self.model.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)
        for nid, emb in zip(node_ids, embeddings):
            self._cache[nid] = emb
        logger.info("Embedding cache ready.")

    def get(self, node_id: str) -> np.ndarray:
        if node_id not in self._cache:
            raise KeyError(f"No cached embedding for node '{node_id}'. Call precompute() first.")
        return self._cache[node_id]

    def encode_query(self, query: str) -> np.ndarray:
        return self.model.encode(query, normalize_embeddings=True)


# ---------------------------------------------------------------------------
# Scoring function
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Since embeddings are normalised, this reduces to a dot product.
    """
    return float(np.dot(a, b))


def trajectory_score(
    sim: float,
    degree: int,
    alpha: float = 1.0,
    beta: float = 0.35,
) -> float:
    """
    S(v) = alpha * Sim(e_q, e_v) - beta * ln(Deg(v) + 1)

    The +1 offset ensures well-definedness for leaf nodes (Deg=0 -> ln(1)=0).
    Since the function is used comparatively (argmax), absolute scale of
    alpha does not affect correctness — only rank order matters.

    Parameters
    ----------
    sim    : cosine similarity between query and candidate node embeddings
    degree : out-degree of the candidate node
    alpha  : semantic alignment weight (default 1.0)
    beta   : structural penalization weight (default 0.35)
    """
    return alpha * sim - beta * math.log(degree + 1)


# ---------------------------------------------------------------------------
# HG-STO traversal (Algorithm 1)
# ---------------------------------------------------------------------------

class HGSTO:
    """
    Heuristic-Guided Subgraph Trajectory Optimization traversal engine.

    Parameters
    ----------
    graph     : KnowledgeGraph instance
    cache     : precomputed EmbeddingCache
    alpha     : semantic weight (default 1.0)
    beta      : structural penalization weight (default 0.35)
    K         : maximum traversal horizon (default 3)
    tau       : entity linking similarity threshold (default 0.85)
    """

    def __init__(
        self,
        graph: KnowledgeGraph,
        cache: EmbeddingCache,
        alpha: float = 1.0,
        beta: float = 0.35,
        K: int = 3,
        tau: float = 0.85,
    ):
        self.graph = graph
        self.cache = cache
        self.alpha = alpha
        self.beta = beta
        self.K = K
        self.tau = tau

    def traverse(self, query: str) -> TraversalResult:
        """
        Execute Algorithm 1: HG-STO Traversal.

        Steps
        -----
        1.  Encode query -> e_q
        2.  Entity link query -> seed node v_0
        3.  Initialise trajectory Π, visited set H, context C
        4.  For t = 0..K-1:
              a. Compute unvisited action space A_t
              b. If A_t empty: break (dead-end)
              c. Score each candidate with S(v)
              d. Select v_{t+1} = argmax S(v)
              e. Update Π, H, C
        5.  Return Π, C
        """
        # Step 1: encode query
        e_q = self.cache.encode_query(query)

        # Step 2: entity linking
        seed = fuzzy_entity_link(query, list(self.graph.nodes.keys()), self.tau)
        if seed is None:
            raise ValueError(f"Entity linking failed for query: '{query}'")
        logger.info("Seed node: %s", seed)

        # Step 3: initialise
        trajectory = [seed]
        visited = {seed}
        context = [self.graph.nodes[seed]]
        hop_scores = []

        # Step 4: traverse up to K hops
        for t in range(self.K):
            current = trajectory[-1]
            neighbors = self.graph.neighbors(current)

            # Action space: unvisited neighbors
            action_space = [n for n in neighbors if n not in visited]

            # Step 4b: dead-end
            if not action_space:
                logger.info("Dead-end at hop %d (node: %s). Stopping.", t, current)
                break

            # Step 4c: score each candidate
            step_scores = []
            for candidate in action_space:
                e_v = self.cache.get(candidate)
                sim = cosine_similarity(e_q, e_v)
                deg = self.graph.out_degree(candidate)
                penalty = self.beta * math.log(deg + 1)
                score = self.alpha * sim - penalty
                step_scores.append({
                    "node": candidate,
                    "sim": round(sim, 4),
                    "degree": deg,
                    "penalty": round(penalty, 4),
                    "score": round(score, 4),
                })

            # Step 4d: select best candidate
            best = max(step_scores, key=lambda x: x["score"])
            hop_scores.append({"hop": t + 1, "candidates": step_scores, "selected": best["node"]})
            logger.info(
                "Hop %d: selected '%s' (score=%.4f, sim=%.4f, deg=%d, penalty=%.4f)",
                t + 1, best["node"], best["score"], best["sim"], best["degree"], best["penalty"]
            )

            # Step 4e: update
            next_node = best["node"]
            trajectory.append(next_node)
            visited.add(next_node)
            context.append(self.graph.nodes[next_node])

        return TraversalResult(
            trajectory=trajectory,
            context=context,
            hop_scores=hop_scores,
            seed_node=seed,
            query=query,
        )
