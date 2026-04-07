"""
Fetches a large corpus of real Clash Royale decks from top-ladder players
for training Card2Vec embeddings.

Strategy:
  1. Fetch the top N global players by trophy count from the leaderboard
  2. For each player, fetch their recent battle log
  3. Extract all decks from every battle (both player and opponent sides)
  4. Deduplicate and return as a flat list of card-name lists

Why both player and opponent decks?
  Word2Vec / PMI learns from co-occurrence patterns, not win rates.
  A Golem Night Witch deck is a Golem Night Witch deck whether it won or lost.
  More decks = better co-occurrence statistics = better embeddings.
  The meta weighter already handles win-rate analysis separately.

Rate limiting:
  The CR API is rate-limited. We fetch players in small batches with a short
  delay between batches to avoid 429 errors. The training script is meant to
  be run offline (not on every request), so latency is acceptable.
"""
import asyncio
import httpx

BASE_URL = "https://proxy.royaleapi.dev/v1"
GLOBAL_LOCATION_ID = 57000249


async def fetch_top_decks(
    api_key: str,
    n_players: int = 100,
    battles_per_player: int = 25,
    batch_size: int = 10,
    batch_delay: float = 0.5,
) -> list[list[str]]:
    """
    Fetch a large corpus of real decks from top-ladder players.

    Args:
        api_key: Clash Royale API key (Bearer token)
        n_players: how many top players to sample (default 100)
        battles_per_player: battle log limit per player (default 25)
        batch_size: concurrent requests per batch (default 10)
        batch_delay: seconds to sleep between batches (default 0.5)

    Returns:
        Deduplicated list of decks, each deck a list of 8 lowercased card names.
        Example: [["hog rider", "musketeer", "fireball", ...], ...]
    """
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers=headers,
        timeout=30.0,
    ) as client:
        # Step 1: Get top player tags
        tags = await _fetch_top_player_tags(client, n_players)
        print(f"  Fetched {len(tags)} top player tags from leaderboard")

        if not tags:
            print("  Warning: no player tags fetched — check API key and connectivity")
            return []

        # Step 2: Fetch battle logs in batches
        all_decks: list[list[str]] = []
        total_batches = (len(tags) + batch_size - 1) // batch_size

        for batch_num, i in enumerate(range(0, len(tags), batch_size), start=1):
            batch = tags[i : i + batch_size]
            print(f"  Batch {batch_num}/{total_batches}: fetching {len(batch)} players...")

            results = await asyncio.gather(
                *[_fetch_player_decks(client, tag, battles_per_player) for tag in batch],
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, list):
                    all_decks.extend(result)
                # Silently skip exceptions — one failed player shouldn't abort training

            # Respect rate limits between batches
            if i + batch_size < len(tags):
                await asyncio.sleep(batch_delay)

        print(f"  Collected {len(all_decks)} raw decks")

    # Step 3: Deduplicate by card set
    # Two decks are the same if they contain the same 8 cards (order doesn't matter)
    seen: set[frozenset[str]] = set()
    unique_decks: list[list[str]] = []

    for deck in all_decks:
        key = frozenset(deck)
        if key not in seen and len(deck) == 8:
            seen.add(key)
            unique_decks.append(deck)

    print(f"  After deduplication: {len(unique_decks)} unique decks")
    return unique_decks


async def _fetch_top_player_tags(
    client: httpx.AsyncClient,
    n: int,
) -> list[str]:
    """
    Fetch the top N player tags from the global leaderboard.
    Returns a list of tag strings (without the '#' prefix).
    """
    try:
        resp = await client.get(
            f"/locations/{GLOBAL_LOCATION_ID}/rankings/players",
            params={"limit": min(n, 1000)},
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        return [item["tag"].lstrip("#") for item in items[:n]]
    except httpx.HTTPStatusError as e:
        print(f"  Leaderboard fetch failed ({e.response.status_code}): {e}")
        return []
    except Exception as e:
        print(f"  Leaderboard fetch failed: {e}")
        return []


async def _fetch_player_decks(
    client: httpx.AsyncClient,
    player_tag: str,
    n_battles: int,
) -> list[list[str]]:
    """
    Fetch all decks (player + opponent) from a player's recent battle log.

    Returns a list of decks, each deck a list of 8 lowercased card name strings.
    Returns an empty list on any error so one bad player doesn't abort the batch.
    """
    try:
        tag = player_tag.strip().lstrip("#")
        resp = await client.get(
            f"/players/%23{tag}/battlelog",
            params={"limit": n_battles},
        )
        resp.raise_for_status()
        battles = resp.json()

        decks: list[list[str]] = []
        for battle in battles:
            for side_key in ("team", "opponent"):
                for side in battle.get(side_key, []):
                    cards = side.get("cards", [])
                    if len(cards) == 8:
                        deck = [c["name"].lower() for c in cards]
                        decks.append(deck)

        return decks

    except Exception:
        # Silent failure — one player's missing battle log shouldn't stop training
        return []
