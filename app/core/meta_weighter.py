from collections import defaultdict
from dataclasses import dataclass
from app.core.models import VulnerabilityProfile, DamageType, TargetType


@dataclass
class ArenaMetaWeights:
    """
    Represents how dangerous each threat type is in a specific arena,
    derived from real battle data at that trophy range.

    The intuition: if 60% of decks you face run a tank like Golem,
    your tank_exposure score should be weighted much more heavily
    than if only 10% run tanks. This mirrors how a quant weights
    factor exposures by their probability of occurring.
    """
    arena_name: str
    trophy_range: tuple[int, int]
    air_threat_rate: float = 0.0       # % of decks with air win conditions
    tank_threat_rate: float = 0.0      # % of decks with tank cards
    swarm_threat_rate: float = 0.0     # % of decks with swarm cards
    spell_threat_rate: float = 0.0     # % of decks with splash spells
    building_threat_rate: float = 0.0  # % of decks with defensive buildings
    high_elixir_rate: float = 0.0      # % of decks with avg elixir > 4.0
    sample_size: int = 0               # how many battles this is based on

    def as_weight_vector(self) -> dict[str, float]:
        """
        Convert threat rates into a normalized weight vector.
        The weights tell the optimizer how much each vulnerability
        factor matters in this specific arena.

        We add a baseline of 0.1 so no factor ever gets zero weight —
        even rare threats should have some consideration.
        """
        raw = {
            "air": self.air_threat_rate + 0.1,
            "tank": self.tank_threat_rate + 0.1,
            "swarm": self.swarm_threat_rate + 0.1,
            "spell": self.spell_threat_rate + 0.1,
            "building": self.building_threat_rate + 0.1,
            "elixir": self.high_elixir_rate + 0.1,
        }
        # Normalize so weights sum to 1 — same as portfolio weight normalization
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}


# Cards that are air-based win conditions
AIR_WIN_CON_NAMES = {"balloon", "lava hound", "minions", "minion horde"}

# Cards that are tanks
TANK_NAMES = {"giant", "golem", "pekka", "mega knight", "electro giant",
              "goblin giant", "royal giant", "lava hound"}

# Swarm cards
SWARM_NAMES = {"goblin gang", "skeleton army", "minion horde", "goblins",
               "archers", "minions", "skeletons", "rascals", "spear goblins",
               "barbarians"}

# Splash spells
SPLASH_SPELL_NAMES = {"fireball", "arrows", "the log", "poison", "lightning",
                      "zap", "earthquake", "rocket", "tornado", "giant snowball",
                      "barbarian barrel", "royal delivery"}

# Defensive buildings
DEFENSIVE_BUILDING_NAMES = {"cannon", "tesla", "inferno tower", "bomb tower",
                             "goblin cage", "tombstone", "electro spirit",
                             "mortar", "x-bow"}


def compute_meta_weights(battles: list[dict], player_tag: str) -> ArenaMetaWeights:
    """
    Analyze a player's battle log to compute arena meta weights.

    We look at OPPONENT decks only — the goal is to understand
    what threats you'll face, not what you're already running.

    Each battle gives us one opponent deck = 8 cards.
    We count how often each threat archetype appears across all battles.
    """
    if not battles:
        return _default_weights()

    # Get arena info from first battle
    arena_name = battles[0].get("arena", {}).get("name", "Unknown")
    trophies = battles[0].get("team", [{}])[0].get("startingTrophies", 0)
    trophy_range = (_bucket_trophies(trophies))

    # Counters for each threat type
    threat_counts = defaultdict(int)
    total_battles = 0

    for battle in battles:
        # Only look at opponent decks
        opponents = battle.get("opponent", [])
        if not opponents:
            continue

        opponent_cards = opponents[0].get("cards", [])
        if not opponent_cards:
            continue

        card_names = {c["name"].lower() for c in opponent_cards}
        elixir_costs = [c.get("elixirCost", 0) for c in opponent_cards]
        avg_elixir = sum(elixir_costs) / len(elixir_costs) if elixir_costs else 0

        # Check each threat type
        if card_names & AIR_WIN_CON_NAMES:
            threat_counts["air"] += 1
        if card_names & TANK_NAMES:
            threat_counts["tank"] += 1
        if card_names & SWARM_NAMES:
            threat_counts["swarm"] += 1
        if card_names & SPLASH_SPELL_NAMES:
            threat_counts["spell"] += 1
        if card_names & DEFENSIVE_BUILDING_NAMES:
            threat_counts["building"] += 1
        if avg_elixir > 4.0:
            threat_counts["elixir"] += 1

        total_battles += 1

    if total_battles == 0:
        return _default_weights()

    # Convert counts to rates
    return ArenaMetaWeights(
        arena_name=arena_name,
        trophy_range=trophy_range,
        air_threat_rate=threat_counts["air"] / total_battles,
        tank_threat_rate=threat_counts["tank"] / total_battles,
        swarm_threat_rate=threat_counts["swarm"] / total_battles,
        spell_threat_rate=threat_counts["spell"] / total_battles,
        building_threat_rate=threat_counts["building"] / total_battles,
        high_elixir_rate=threat_counts["elixir"] / total_battles,
        sample_size=total_battles,
    )


def apply_meta_weights(
    profile: VulnerabilityProfile,
    weights: ArenaMetaWeights
) -> float:
    """
    Apply arena meta weights to a vulnerability profile to get a
    single weighted score.

    This is the core of our objective function — analogous to computing
    a portfolio's weighted factor exposure:
        score = sum(factor_exposure_i * factor_weight_i)

    Lower score = better optimized deck for this arena.
    We subtract win_condition_score because offensive strength
    offsets vulnerability — a deck that can punish mistakes
    is less risky even if it has some exposures.
    """
    w = weights.as_weight_vector()

    weighted_score = (
        profile.air_exposure      * w["air"] +
        profile.tank_exposure     * w["tank"] +
        profile.swarm_exposure    * w["swarm"] +
        profile.spell_exposure    * w["spell"] +
        profile.building_exposure * w["building"] +
        profile.elixir_risk       * w["elixir"]
    )

    # Win condition offsets vulnerability — strong offense = lower net risk
    win_con_offset = profile.win_condition_score * 0.2

    return max(0.0, weighted_score - win_con_offset)


def _bucket_trophies(trophies: int) -> tuple[int, int]:
    """Map a trophy count to a range bucket for display purposes."""
    buckets = [
        (0, 1000), (1000, 2000), (2000, 3000), (3000, 4000),
        (4000, 5000), (5000, 6000), (6000, 7000), (7000, 8000),
        (8000, 9000), (9000, 10000), (10000, 12000), (12000, 99999)
    ]
    for low, high in buckets:
        if low <= trophies < high:
            return (low, high)
    return (0, 99999)


def _default_weights() -> ArenaMetaWeights:
    """Equal weights fallback when no battle data is available."""
    return ArenaMetaWeights(
        arena_name="Unknown",
        trophy_range=(0, 99999),
        air_threat_rate=0.5,
        tank_threat_rate=0.5,
        swarm_threat_rate=0.5,
        spell_threat_rate=0.5,
        building_threat_rate=0.5,
        high_elixir_rate=0.5,
        sample_size=0,
    )