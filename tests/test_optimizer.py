import asyncio
from app.data.client import ClashRoyaleClient
from app.data.card_loader import get_card_registry
from app.core.models import Deck
from app.core.analyzer import DeckAnalyzer
from app.core.meta_weighter import compute_meta_weights, apply_meta_weights
from app.core.optimizer import DeckOptimizer

async def test():
    client = ClashRoyaleClient()

    # Fetch your real current deck and battle log
    player = await client.get_player("9CGPRG09")
    resp = await client.client.get(
        "/players/%239CGPRG09/battlelog",
        params={"limit": 25}
    )
    resp.raise_for_status()
    battles = resp.json()
    await client.close()

    # Build meta weights from your battle log
    weights = compute_meta_weights(battles, "9CGPRG09")
    print(f"Arena: {weights.arena_name} ({weights.sample_size} battles)")

    # Get your current deck from your profile
    card_registry = get_card_registry()
    current_card_names = [c['name'].lower() for c in player['currentDeck']]
    print(f"\nYour current deck: {[c['name'] for c in player['currentDeck']]}")

    # Map to our Card objects
    deck_cards = []
    for name in current_card_names:
        card = card_registry.get(name)
        if card:
            deck_cards.append(card)
        else:
            print(f"  Warning: '{name}' not found in registry")

    if len(deck_cards) != 8:
        print(f"Only matched {len(deck_cards)}/8 cards, skipping optimization")
        return

    deck = Deck(cards=deck_cards)

    # Analyze current deck
    analyzer = DeckAnalyzer()
    profile = analyzer.analyze(deck)
    score = apply_meta_weights(profile, weights)

    print(f"\nCurrent deck analysis:")
    print(f"  Avg elixir:        {deck.average_elixir:.2f}")
    print(f"  Air exposure:      {profile.air_exposure:.2f}")
    print(f"  Tank exposure:     {profile.tank_exposure:.2f}")
    print(f"  Swarm exposure:    {profile.swarm_exposure:.2f}")
    print(f"  Spell exposure:    {profile.spell_exposure:.2f}")
    print(f"  Building exposure: {profile.building_exposure:.2f}")
    print(f"  Elixir risk:       {profile.elixir_risk:.2f}")
    print(f"  Win con score:     {profile.win_condition_score:.2f}")
    print(f"  Weighted score:    {score:.3f}")

    # Run optimizer
    optimizer = DeckOptimizer()
    result = optimizer.optimize(deck, card_registry, weights)

    print(f"\n--- Optimization Results ---")
    if result.best_swap:
        print(f"\nBest swap:")
        print(f"  {optimizer.explain_swap(result.best_swap)}")

        print(f"\nAlternative swaps:")
        for swap in result.alternatives:
            print(f"  {optimizer.explain_swap(swap)}")
    else:
        print("No improvements found — your deck is already well optimized!")

asyncio.run(test())