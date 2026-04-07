"""
DeckCohesionScorer: scores how well a candidate card fits an existing deck
using pre-trained Card2Vec embeddings.

The core idea:
  When the optimizer tries swapping card_out for card_in, we ask:
  "Does card_in belong in a deck with these other 7 cards?"

  We answer by computing cosine similarity between:
    - card_in's embedding vector
    - the centroid (mean) of the remaining 7 cards' embedding vectors

  High similarity (near 1.0) → card_in typically appears in the same archetype
  Low similarity (near 0.0) → card_in is from a different archetype family

Example:
  Deck centroid = [Hog Rider, Musketeer, Fireball, ...]  (fast-cycle aggro)
  card_in = Ice Golem     → high cohesion (fits aggro cycle perfectly)
  card_in = Graveyard     → low cohesion (wrong archetype — win-condition spell)
  card_in = Night Witch   → medium cohesion (beatdown support, not ideal here)

Graceful degradation:
  If card_embeddings.json doesn't exist yet (embeddings not trained), all scores
  return 0.5 (neutral). The optimizer continues working via role gates and
  affinity bonus — the ML layer simply isn't active until you run train_embeddings.py.
"""
from __future__ import annotations
import numpy as np
from app.ml.card2vec import load_embeddings


class DeckCohesionScorer:
    """
    Loads pre-trained card embeddings and scores deck cohesion for the optimizer.

    Instantiated once at app startup (in lifespan) and shared across requests.
    The embeddings dict is read-only after loading — fully thread-safe.
    """

    def __init__(self, embeddings: dict[str, list[float]] | None = None):
        """
        Args:
            embeddings: pre-loaded embeddings dict (card_name -> vector).
                        If None, loads from app/ml/card_embeddings.json.
                        Pass an empty dict to disable ML scoring.
        """
        raw = embeddings if embeddings is not None else load_embeddings()

        # Store as numpy arrays, keyed by lowercase card name
        self._embeddings: dict[str, np.ndarray] = {
            name.lower(): np.array(vec, dtype=np.float32)
            for name, vec in raw.items()
        }
        self._available = len(self._embeddings) > 0

        if self._available:
            sample_dim = next(iter(self._embeddings.values())).shape[0]
            print(
                f"[CohesionScorer] Loaded {len(self._embeddings)} card embeddings "
                f"(dim={sample_dim})"
            )
        else:
            print(
                "[CohesionScorer] No embeddings found — cohesion scoring disabled. "
                "Run train_embeddings.py to enable ML game-feel."
            )

    @property
    def is_available(self) -> bool:
        """True if embeddings are loaded and cohesion scoring is active."""
        return self._available

    def score(
        self,
        card_in_name: str,
        remaining_card_names: list[str],
    ) -> float:
        """
        Score how well card_in fits with the remaining deck cards.

        Returns a float in [0.0, 1.0]:
          1.0 → card_in consistently appears in decks like this (great fit)
          0.5 → no data available (neutral — no penalty, no bonus)
          0.0 → card_in is from a completely different archetype (poor fit)

        The 0.5 neutral default is intentional: when we lack embedding data
        for a card (e.g. a very new card or one not in the training corpus),
        we don't want to penalize it — we just don't boost it.

        Args:
            card_in_name: the candidate card being swapped in
            remaining_card_names: the 7 cards staying in the deck
                                   (card_out has already been excluded)

        Returns:
            cohesion score in [0.0, 1.0]
        """
        if not self._available:
            return 0.5

        card_in_vec = self._embeddings.get(card_in_name.lower())
        if card_in_vec is None:
            return 0.5  # unknown card — neutral

        # Gather embeddings for the remaining deck cards
        context_vecs = [
            self._embeddings[name.lower()]
            for name in remaining_card_names
            if name.lower() in self._embeddings
        ]

        if not context_vecs:
            return 0.5  # no context embeddings — neutral

        # Compute deck centroid and normalize it
        centroid = np.mean(context_vecs, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm == 0:
            return 0.5
        centroid = centroid / centroid_norm

        # card_in_vec is already L2-normalized from training,
        # so dot product = cosine similarity.
        similarity = float(np.dot(card_in_vec, centroid))

        # Clamp to [0, 1]:
        # Cosine similarity over PPMI embeddings is always >= 0 in practice
        # (PPMI is non-negative, so SVD vectors tend to have non-negative dot products),
        # but we clamp defensively.
        return max(0.0, min(1.0, similarity))

    def nearest_neighbors(
        self,
        card_name: str,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """
        Return the top-N most similar cards to card_name by cosine similarity.
        Useful for debugging and validating embedding quality.

        Example output for 'hog rider':
          [('musketeer', 0.94), ('ice golem', 0.91), ('fireball', 0.89), ...]
        """
        target_vec = self._embeddings.get(card_name.lower())
        if target_vec is None:
            return []

        similarities = [
            (name, float(np.dot(target_vec, vec)))
            for name, vec in self._embeddings.items()
            if name != card_name.lower()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
