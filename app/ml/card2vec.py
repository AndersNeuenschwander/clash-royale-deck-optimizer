"""
Card2Vec: learns card embeddings from deck co-occurrence patterns.

We use PMI (Pointwise Mutual Information) + truncated SVD — mathematically
equivalent to Word2Vec but simpler to implement with just numpy.

The intuition:
  - Cards that consistently appear together in real decks (e.g. Hog Rider +
    Musketeer, or Golem + Night Witch) end up with nearby embedding vectors.
  - Cards from different archetypes (e.g. Graveyard vs. Knight) end up far apart.
  - When the optimizer evaluates a swap, we compute cosine similarity between
    card_in's embedding and the centroid of the remaining 7 cards. Low similarity
    means the candidate card comes from a different deck archetype — penalize it.

Why PMI + SVD instead of neural Word2Vec?
  - No training loop, no hyperparameter tuning, no framework dependency.
  - SVD acts as a built-in denoising step — it discards noise from sparse
    co-occurrence counts and extracts the dominant co-occurrence patterns.
  - Same theoretical foundation: both PMI and skip-gram Word2Vec learn to
    predict context from target (the Levy & Goldberg 2014 equivalence result).
  - Fast: a 120×120 matrix SVD takes milliseconds.
"""
import json
import os
import numpy as np

EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "card_embeddings.json")
EMBEDDING_DIM = 32


def train_embeddings(
    decks: list[list[str]],
    embedding_dim: int = EMBEDDING_DIM,
) -> dict[str, list[float]]:
    """
    Train card embeddings from a corpus of decks using PMI + SVD.

    Each deck is treated as a co-occurrence context: every pair of cards
    in the same deck contributes to their mutual count. This is analogous
    to Word2Vec with a window size equal to the full deck length.

    Args:
        decks: list of decks, each deck is a list of card name strings
               (lowercased, e.g. ["hog rider", "musketeer", "fireball", ...])
        embedding_dim: dimensionality of output embeddings (default 32)

    Returns:
        dict mapping card_name (str) -> embedding vector (list[float])
        All vectors are L2-normalized so cosine similarity = dot product.
    """
    if not decks:
        return {}

    # --- Step 1: Build vocabulary ---
    vocab = sorted({card for deck in decks for card in deck})
    card_to_idx = {card: i for i, card in enumerate(vocab)}
    V = len(vocab)

    if V < 2:
        return {}

    # --- Step 2: Build co-occurrence matrix ---
    # M[i][j] = number of decks where card i and card j both appear.
    # We count all ordered pairs (i, j) and (j, i) so the matrix is symmetric.
    M = np.zeros((V, V), dtype=np.float64)

    for deck in decks:
        # Deduplicate within a deck (shouldn't happen in CR but be safe)
        unique_cards = list({c for c in deck if c in card_to_idx})
        for card_a in unique_cards:
            for card_b in unique_cards:
                if card_a != card_b:
                    M[card_to_idx[card_a], card_to_idx[card_b]] += 1.0

    # --- Step 3: Compute PPMI (Positive Pointwise Mutual Information) ---
    #
    # PMI(i, j) = log[ P(i,j) / (P(i) * P(j)) ]
    #           = log[ M[i,j] * total / (row_sum[i] * col_sum[j]) ]
    #
    # PPMI clips negative values to 0 — negative co-occurrence is unreliable
    # with small corpora and adds noise to embeddings.
    #
    # The intuition: if two cards co-occur MORE than chance predicts,
    # they have a positive PPMI value — they "belong" together.
    # If they co-occur at chance level or less, PPMI = 0.

    total = M.sum()
    if total == 0:
        return {}

    row_sums = M.sum(axis=1, keepdims=True)   # shape (V, 1)
    col_sums = M.sum(axis=0, keepdims=True)   # shape (1, V)
    expected = (row_sums @ col_sums) / total   # shape (V, V)

    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.where(
            (M > 0) & (expected > 0),
            np.log(M / expected),
            0.0,
        )
    ppmi = np.maximum(0.0, pmi)

    # --- Step 4: Truncated SVD ---
    #
    # Decompose: PPMI ≈ U * diag(S) * Vt
    # Embeddings: U[:, :k] * sqrt(S[:k])
    #
    # This is the "eigenword" formula. Scaling by sqrt(S) ensures that
    # dimensions with more variance (stronger co-occurrence signal) contribute
    # more to cosine similarity — higher-confidence patterns dominate.
    #
    # We take the minimum of embedding_dim and V-1 to handle small vocabs.

    k = min(embedding_dim, V - 1)
    U, S, _Vt = np.linalg.svd(ppmi, full_matrices=False)

    embeddings_matrix = U[:, :k] * np.sqrt(S[:k])

    # --- Step 5: L2 normalize rows ---
    # After normalization, cosine_similarity(a, b) = dot(a, b).
    # This makes inference fast: just compute dot products.
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings_matrix = embeddings_matrix / norms

    return {
        vocab[i]: embeddings_matrix[i].tolist()
        for i in range(V)
    }


def save_embeddings(
    embeddings: dict[str, list[float]],
    path: str = EMBEDDINGS_PATH,
) -> None:
    """Save embeddings dict to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(embeddings, f)
    print(f"Saved {len(embeddings)} card embeddings to {path}")


def load_embeddings(path: str = EMBEDDINGS_PATH) -> dict[str, list[float]]:
    """
    Load embeddings from a JSON file.
    Returns an empty dict (graceful degradation) if the file doesn't exist yet.
    The cohesion scorer treats missing embeddings as a neutral 0.5 signal,
    so the optimizer still works — just without the ML layer.
    """
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)
