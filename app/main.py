from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from copy import copy
from app.data.client import ClashRoyaleClient
from app.data.card_loader import get_card_registry
from app.core.analyzer import DeckAnalyzer
from app.core.meta_weighter import compute_meta_weights, apply_meta_weights
from app.core.optimizer import DeckOptimizer
from app.core.models import Deck
from app.db.database import engine
from app.db import models
from app.api import auth, user_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = ClashRoyaleClient()
    app.state.card_registry = get_card_registry()
    app.state.analyzer = DeckAnalyzer()
    app.state.optimizer = DeckOptimizer()
    yield
    await app.state.client.close()


app = FastAPI(
    title="Clash Royale Deck Optimizer",
    description="Analyze and optimize your Clash Royale deck based on your arena meta",
    version="0.1.0",
    lifespan=lifespan
)

# Create all database tables on startup
models.Base.metadata.create_all(bind=engine)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the frontend from /static
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Register routers
app.include_router(auth.router)
app.include_router(user_data.router)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Clash Royale Deck Optimizer API"}


@app.get("/cards")
async def get_all_cards():
    """Returns icon URLs for all cards, used by the frontend to display swap suggestions."""
    client: ClashRoyaleClient = app.state.client
    data = await client.get_cards()
    return {
        item["name"].lower(): item.get("iconUrls", {}).get("medium")
        for item in data.get("items", [])
    }


@app.get("/player/{player_tag}")
async def get_player_analysis(player_tag: str):
    client: ClashRoyaleClient = app.state.client
    card_registry = app.state.card_registry
    analyzer = app.state.analyzer
    optimizer = app.state.optimizer

    tag = player_tag.strip().lstrip("#").upper()

    try:
        import asyncio
        player_data, battles_resp = await asyncio.gather(
            client.get_player(tag),
            client.client.get(
                f"/players/%23{tag}/battlelog",
                params={"limit": 25}
            )
        )
        battles_resp.raise_for_status()
        battles = battles_resp.json()

    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Player not found: {str(e)}")

    weights = compute_meta_weights(battles, tag)

    current_deck_raw = player_data.get("currentDeck", [])
    deck_cards = []
    unmatched = []

    for card_data in current_deck_raw:
        name = card_data["name"].lower()
        card = card_registry.get(name)
        if card:
            card = copy(card)
            card.icon_url = card_data.get("iconUrls", {}).get("medium")
            deck_cards.append(card)
        else:
            unmatched.append(card_data["name"])

    if len(deck_cards) < 8:
        raise HTTPException(
            status_code=422,
            detail=f"Could only match {len(deck_cards)}/8 cards. Unmatched: {unmatched}"
        )

    deck = Deck(cards=deck_cards)

    profile = analyzer.analyze(deck)
    weighted_score = apply_meta_weights(profile, weights)
    result = optimizer.optimize(deck, card_registry, weights)

    def format_swap(swap):
        return {
            "card_out": {
                "name": swap.card_out.name,
                "elixir": swap.card_out.elixir_cost,
                "rarity": swap.card_out.rarity.value,
                "icon_url": swap.card_out.icon_url,
            },
            "card_in": {
                "name": swap.card_in.name,
                "elixir": swap.card_in.elixir_cost,
                "rarity": swap.card_in.rarity.value,
                "icon_url": swap.card_in.icon_url,
            },
            "score_before": round(swap.score_before, 3),
            "score_after": round(swap.score_after, 3),
            "improvement_pct": round(swap.improvement_pct, 1),
            "explanation": optimizer.explain_swap(swap),
        }

    swaps = []
    if result.best_swap:
        swaps.append(format_swap(result.best_swap))
    swaps.extend([format_swap(s) for s in result.alternatives])

    return {
        "player": {
            "tag": player_data.get("tag"),
            "name": player_data.get("name"),
            "trophies": player_data.get("trophies"),
            "arena": player_data.get("arena", {}).get("name"),
        },
        "current_deck": [
            {
                "name": c.name,
                "elixir": c.elixir_cost,
                "rarity": c.rarity.value,
                "icon_url": c.icon_url,
            }
            for c in deck_cards
        ],
        "analysis": {
            "average_elixir": round(deck.average_elixir, 2),
            "weighted_score": round(weighted_score, 3),
            "air_exposure": round(profile.air_exposure, 2),
            "tank_exposure": round(profile.tank_exposure, 2),
            "swarm_exposure": round(profile.swarm_exposure, 2),
            "spell_exposure": round(profile.spell_exposure, 2),
            "building_exposure": round(profile.building_exposure, 2),
            "elixir_risk": round(profile.elixir_risk, 2),
            "win_condition_score": round(profile.win_condition_score, 2),
        },
        "meta": {
            "arena": weights.arena_name,
            "trophy_range": weights.trophy_range,
            "sample_size": weights.sample_size,
            "threat_rates": {
                "air": round(weights.air_threat_rate, 2),
                "tank": round(weights.tank_threat_rate, 2),
                "swarm": round(weights.swarm_threat_rate, 2),
                "spell": round(weights.spell_threat_rate, 2),
                "building": round(weights.building_threat_rate, 2),
                "high_elixir": round(weights.high_elixir_rate, 2),
            },
            "weights": {k: round(v, 3) for k, v in weights.as_weight_vector().items()},
        },
        "suggestions": swaps,
    }