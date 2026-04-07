#!/usr/bin/env python3
"""
Train and save Card2Vec embeddings for the Clash Royale Deck Optimizer.

This script fetches top-ladder decks from the CR API, trains card embeddings
using PMI + SVD, and saves them to app/ml/card_embeddings.json.

Run this once before deploying, then re-run periodically (e.g. weekly) to
keep the embeddings fresh as the meta evolves.

Usage:
    python train_embeddings.py

    # Faster run with fewer players (lower quality, useful for testing):
    python train_embeddings.py --players 20

    # More data for better embeddings:
    python train_embeddings.py --players 200

Environment:
    CLASH_ROYALE_API_KEY  — required, your Supercell developer API token

Output:
    app/ml/card_embeddings.json  — the trained embeddings file
"""
import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Card2Vec embeddings from top-ladder CR battle logs"
    )
    parser.add_argument(
        "--players",
        type=int,
        default=100,
        help="Number of top players to sample battle logs from (default: 100)",
    )
    parser.add_argument(
        "--battles",
        type=int,
        default=25,
        help="Battle log limit per player (default: 25)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=32,
        help="Embedding dimensionality (default: 32)",
    )
    return parser.parse_args()


async def main(n_players: int, battles_per_player: int, embedding_dim: int) -> None:
    api_key = os.getenv("CLASH_ROYALE_API_KEY")
    if not api_key:
        print("Error: CLASH_ROYALE_API_KEY is not set.")
        print("Set it in your .env file or export it as an environment variable.")
        sys.exit(1)

    from app.data.deck_fetcher import fetch_top_decks
    from app.ml.card2vec import train_embeddings, save_embeddings

    # --- Step 1: Collect deck corpus ---
    print(f"\nFetching decks from top {n_players} players ({battles_per_player} battles each)...")
    decks = await fetch_top_decks(
        api_key=api_key,
        n_players=n_players,
        battles_per_player=battles_per_player,
    )

    if len(decks) < 50:
        print(
            f"\nWarning: only {len(decks)} unique decks collected. "
            "Embeddings may be low quality with sparse data. "
            "Try increasing --players or check your API key."
        )
    else:
        print(f"\nCollected {len(decks)} unique decks — ready to train.")

    # --- Step 2: Train embeddings ---
    print(f"Training Card2Vec embeddings (dim={embedding_dim})...")
    embeddings = train_embeddings(decks, embedding_dim=embedding_dim)

    if not embeddings:
        print("Error: training produced no embeddings. Aborting.")
        sys.exit(1)

    print(f"Trained embeddings for {len(embeddings)} unique cards.")

    # --- Step 3: Save ---
    save_embeddings(embeddings)

    # --- Step 4: Sanity check ---
    print("\nNearest neighbors sanity check:")
    _sanity_check(embeddings)

    print("\nDone. Run your server and the cohesion scorer will load automatically.")


def _sanity_check(embeddings: dict[str, list[float]]) -> None:
    """
    Print nearest neighbors for iconic cards to verify embedding quality.

    Good embeddings should show:
      hog rider   → musketeer, ice golem, fireball   (fast cycle)
      golem       → night witch, baby dragon, lumberjack  (beatdown)
      graveyard   → poison, skeleton army, miner      (graveyard cycle)
      knight      → fireball, musketeer, hog rider    (versatile defensive)
    """
    import numpy as np

    test_cards = ["hog rider", "golem", "graveyard", "knight", "balloon", "x-bow"]
    vecs = {name: np.array(vec, dtype=np.float32) for name, vec in embeddings.items()}

    for target in test_cards:
        if target not in vecs:
            print(f"  {target!r}: not in embeddings (card may not appear in training data)")
            continue

        target_vec = vecs[target]
        similarities = [
            (name, float(np.dot(target_vec, vec)))
            for name, vec in vecs.items()
            if name != target
        ]
        top5 = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]
        neighbors_str = ", ".join(f"{name} ({score:.2f})" for name, score in top5)
        print(f"  {target!r} → {neighbors_str}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            n_players=args.players,
            battles_per_player=args.battles,
            embedding_dim=args.dim,
        )
    )
