import asyncio
import json
from app.data.client import ClashRoyaleClient
from app.core.meta_weighter import compute_meta_weights, apply_meta_weights
from app.core.analyzer import DeckAnalyzer
from app.core.models import Deck
from app.data.card_loader import get_card_registry

async def test():
    client = ClashRoyaleClient()

    resp = await client.client.get(
        "/players/%239CGPRG09/battlelog",
        params={"limit": 25}
    )
    resp.raise_for_status()
    battles = resp.json()

    weights = compute_meta_weights(battles, "9CGPRG09")
    print(f"Arena: {weights.arena_name}")
    print(f"Trophy range: {weights.trophy_range}")
    print(f"Sample size: {weights.sample_size} battles")
    print()
    print("Threat rates in your arena:")
    print(f"  Air threats:      {weights.air_threat_rate:.0%}")
    print(f"  Tank threats:     {weights.tank_threat_rate:.0%}")
    print(f"  Swarm threats:    {weights.swarm_threat_rate:.0%}")
    print(f"  Spell threats:    {weights.spell_threat_rate:.0%}")
    print(f"  Building threats: {weights.building_threat_rate:.0%}")
    print(f"  High elixir:      {weights.high_elixir_rate:.0%}")
    print()
    print("Meta weights:", weights.as_weight_vector())

    await client.close()

asyncio.run(test())