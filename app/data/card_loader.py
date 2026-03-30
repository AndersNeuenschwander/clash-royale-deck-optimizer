import json
import os
from app.core.models import (
    Card, CardAttributes, CardType, Rarity, DamageType, TargetType, CardRole
)
 
DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets")
 
 
def _load_json(filename: str) -> list:
    path = os.path.join(DATASET_DIR, filename)
    with open(path) as f:
        return json.load(f)
 
 
def _parse_rarity(rarity_str: str) -> Rarity:
    mapping = {
        "common": Rarity.COMMON,
        "rare": Rarity.RARE,
        "epic": Rarity.EPIC,
        "legendary": Rarity.LEGENDARY,
        "champion": Rarity.CHAMPION,
    }
    return mapping.get(rarity_str.lower(), Rarity.COMMON)
 
 
def _parse_target_type(attacks_ground: bool, attacks_air: bool) -> TargetType:
    if attacks_ground and attacks_air:
        return TargetType.BOTH
    elif attacks_air:
        return TargetType.AIR
    elif attacks_ground:
        return TargetType.GROUND
    return TargetType.BUILDINGS
 
 
# ---------------------------------------------------------------------------
# Existing structural sets (unchanged)
# ---------------------------------------------------------------------------
 
ALL_SPELL_KEYS = {
    "fireball", "arrows", "rage", "rocket", "goblin-barrel", "freeze",
    "mirror", "lightning", "zap", "poison", "graveyard", "the-log",
    "tornado", "clone", "earthquake", "barbarian-barrel", "giant-snowball",
    "royal-delivery", "party-rocket"
}
 
SPLASH_SPELL_KEYS = {
    "fireball", "arrows", "the-log", "poison", "lightning", "zap",
    "earthquake", "rocket", "goblin-barrel", "graveyard", "tornado",
    "barbarian-barrel", "giant-snowball", "royal-delivery", "party-rocket"
}
 
WIN_CONDITION_KEYS = {
    "hog-rider", "giant", "golem", "balloon", "miner", "ram-rider",
    "royal-giant", "lava-hound", "graveyard", "x-bow", "mortar",
    "goblin-barrel", "three-musketeers", "battle-ram", "wall-breakers",
    "goblin-giant", "electro-giant", "mega-knight", "skeleton-giants",
    "goblin-drill"
}
 
SWARM_KEYS = {
    "goblin-gang", "skeleton-army", "minion-horde", "goblins",
    "archers", "minions", "skeletons", "rascals", "spear-goblins",
    "barbarians", "skeleton-dragons"
}
 
 
# ---------------------------------------------------------------------------
# Role definition sets
#
# The intuition: each set answers the question "which cards fill this role?"
# A card can appear in multiple sets — that's intentional and correct.
# For example, Valkyrie is both a MINI_TANK and a SWARM_CLEAR, which means
# she can validly replace Knight (MINI_TANK) OR Arrows (SWARM_CLEAR).
#
# These are the hand-defined ground truth labels. The RoleRegistry in
# role_registry.py will later use battle data to *confirm and reweight*
# these labels, but never override them entirely — the hand labels are
# the floor, battle data is the ceiling.
# ---------------------------------------------------------------------------
 
# Cards whose primary job is to threaten or destroy the enemy tower.
# Never swap a win condition for a support or cycle card.
WIN_CONDITION_ROLE_KEYS = {
    "hog-rider", "balloon", "miner", "graveyard", "goblin-barrel",
    "ram-rider", "battle-ram", "wall-breakers", "goblin-drill",
    "x-bow", "mortar", "royal-giant",
}
 
# A second offensive threat that forces the opponent to split their defense.
SECONDARY_WIN_CONDITION_ROLE_KEYS = {
    "miner", "battle-ram", "wall-breakers", "dark-prince",
    "prince", "ram-rider", "skeleton-giants",
}
 
# Spells that directly enable or protect the win condition push.
# Log resets Inferno, Poison kills support troops, Zap resets Sparky.
SPELL_SUPPORT_ROLE_KEYS = {
    "fireball", "poison", "the-log", "zap", "arrows", "earthquake",
    "lightning", "freeze", "tornado", "rocket", "giant-snowball",
    "barbarian-barrel", "royal-delivery",
}
 
# Cheap high-HP cards that walk in front of win conditions absorbing hits.
# Key property: low enough elixir to cycle, tanky enough to survive a few hits.
MINI_TANK_ROLE_KEYS = {
    "knight", "ice-golem", "dark-prince", "battle-ram",
    "giant-skeleton", "prince", "ram-rider",
}
 
# High-HP cards that anchor a full beatdown push.
# In beatdown decks these ARE the win condition.
TANK_ROLE_KEYS = {
    "giant", "golem", "pekka", "electro-giant", "goblin-giant",
    "royal-giant", "lava-hound", "mega-knight",
}
 
# Ranged cards that follow behind a tank dealing damage to defenders.
SUPPORT_TROOP_ROLE_KEYS = {
    "musketeer", "witch", "executioner", "baby-dragon", "mega-minion",
    "electro-wizard", "night-witch", "wizard", "three-musketeers",
    "magic-archer", "hunter", "cannon-cart",
}
 
# Cards primarily used to stop incoming pushes rather than launch attacks.
# Note: many cards are BOTH defensive_support and another role — that's fine.
DEFENSIVE_SUPPORT_ROLE_KEYS = {
    "knight", "valkyrie", "mini-pekka", "inferno-dragon", "mega-knight",
    "ice-wizard", "electro-wizard", "dark-prince", "prince",
    "skeleton-army", "guards", "giant-skeleton", "hunter",
}
 
# Cards whose defining job is killing groups of small units.
SWARM_CLEAR_ROLE_KEYS = {
    "valkyrie", "witch", "executioner", "baby-dragon", "wizard",
    "arrows", "the-log", "fireball", "zap", "giant-snowball",
    "barbarian-barrel", "royal-delivery", "mega-knight",
    "dark-prince", "bowler",
}
 
# Cards that can reliably shoot down air threats.
ANTI_AIR_ROLE_KEYS = {
    "mega-minion", "minions", "minion-horde", "archers", "musketeer",
    "inferno-dragon", "electro-dragon", "baby-dragon", "electro-wizard",
    "hunter", "cannon-cart", "three-musketeers", "witch",
    "skeleton-dragons",
}
 
# Cards with high single-target DPS specifically for melting tanks.
TANK_KILLER_ROLE_KEYS = {
    "inferno-tower", "inferno-dragon", "pekka", "mini-pekka",
    "sparky", "hunter", "cannon-cart",
}
 
# Cheap cards (1-3 elixir) whose main job is cycling the rotation.
# If you remove a cycle card without adding another, your average elixir
# jumps and you lose the ability to quickly cycle back to your win condition.
CYCLE_ROLE_KEYS = {
    "skeletons", "ice-spirit", "goblin", "bats", "zap",
    "spear-goblins", "goblins", "knight", "archers",
    "giant-snowball", "royal-delivery",
}
 
# Stationary structures — defensive or offensive.
# Buildings occupy their own deck slot and cannot swap with troops or spells.
BUILDING_ROLE_KEYS = {
    "cannon", "tesla", "inferno-tower", "bomb-tower", "goblin-cage",
    "tombstone", "mortar", "x-bow", "elixir-collector",
    "goblin-hut", "barbarian-hut", "furnace",
}
 
# Cards that continuously produce units over time.
SPAWNER_ROLE_KEYS = {
    "goblin-hut", "barbarian-hut", "furnace", "tombstone",
    "night-witch", "witch",
}
 
# Cards played specifically to punish an overcommitted opponent.
PUNISHMENT_ROLE_KEYS = {
    "skeleton-army", "giant-skeleton", "pekka", "mini-pekka",
    "barbarians", "dark-prince", "prince",
}
 
 
def _assign_roles(card_key: str, elixir: int, hitpoints: int) -> frozenset:
    """
    Build the frozenset of CardRoles for a card by checking membership
    across every role set defined above.
 
    The intuition: this is a multi-label classification. We're not
    picking ONE role per card — we're asking "which roles does this card
    fulfill?" A card like Valkyrie fills three roles simultaneously
    (MINI_TANK, DEFENSIVE_SUPPORT, SWARM_CLEAR), and that's correct —
    she can substitute for Knight OR for Arrows depending on context,
    and the optimizer will prefer whichever swap also improves the
    vulnerability score the most.
 
    The frozenset makes role comparison a single set-intersection call:
        card_out.roles & card_in.roles  →  non-empty means compatible
    """
    roles = set()
 
    if card_key in WIN_CONDITION_ROLE_KEYS:
        roles.add(CardRole.WIN_CONDITION)
    if card_key in SECONDARY_WIN_CONDITION_ROLE_KEYS:
        roles.add(CardRole.SECONDARY_WIN_CONDITION)
    if card_key in SPELL_SUPPORT_ROLE_KEYS:
        roles.add(CardRole.SPELL_SUPPORT)
    if card_key in MINI_TANK_ROLE_KEYS:
        roles.add(CardRole.MINI_TANK)
    if card_key in TANK_ROLE_KEYS:
        roles.add(CardRole.TANK)
    if card_key in SUPPORT_TROOP_ROLE_KEYS:
        roles.add(CardRole.SUPPORT_TROOP)
    if card_key in DEFENSIVE_SUPPORT_ROLE_KEYS:
        roles.add(CardRole.DEFENSIVE_SUPPORT)
    if card_key in SWARM_CLEAR_ROLE_KEYS:
        roles.add(CardRole.SWARM_CLEAR)
    if card_key in ANTI_AIR_ROLE_KEYS:
        roles.add(CardRole.ANTI_AIR)
    if card_key in TANK_KILLER_ROLE_KEYS:
        roles.add(CardRole.TANK_KILLER)
    if card_key in CYCLE_ROLE_KEYS:
        roles.add(CardRole.CYCLE)
    if card_key in BUILDING_ROLE_KEYS:
        roles.add(CardRole.BUILDING)
    if card_key in SPAWNER_ROLE_KEYS:
        roles.add(CardRole.SPAWNER)
    if card_key in PUNISHMENT_ROLE_KEYS:
        roles.add(CardRole.PUNISHMENT)
 
    # Stat-based fallback: if a card wasn't hand-tagged at all,
    # infer a rough role from its stats so it still participates
    # in swaps rather than being silently excluded.
    if not roles:
        if elixir <= 3:
            roles.add(CardRole.CYCLE)
        elif hitpoints >= 1500:
            roles.add(CardRole.MINI_TANK)
        else:
            roles.add(CardRole.SUPPORT_TROOP)
 
    return frozenset(roles)
 
 
def _parse_damage_type(card_data: dict, card_key: str = "") -> DamageType:
    """
    We check card_key against ALL_SPELL_KEYS to identify spells,
    since the stats dict doesn't carry a 'type' field.
    """
    if card_key in ALL_SPELL_KEYS:
        return DamageType.SPLASH if card_key in SPLASH_SPELL_KEYS else DamageType.SINGLE
 
    if card_data.get("area_damage_radius", 0) > 0:
        return DamageType.SPLASH
    elif card_data.get("multiple_targets", 0) > 0:
        return DamageType.CHAIN
    return DamageType.SINGLE
 
 
def _build_attributes(card_base: dict, stats: dict) -> CardAttributes:
    """
    card_base comes from cards.json (elixir, type, rarity).
    stats comes from cards_stats_*.json (hitpoints, targeting, damage type).
    We merge them here so each source contributes what it's best at.
    """
    card_key = card_base.get("key", "")
    card_type_str = card_base.get("type", "Troop").lower()
    elixir = card_base.get("elixir", 0)
    hitpoints = stats.get("hitpoints", 0)
 
    if card_type_str == "spell":
        card_type = CardType.SPELL
    elif card_type_str == "building":
        card_type = CardType.BUILDING
    else:
        card_type = CardType.TROOP
 
    attacks_ground = stats.get("attacks_ground", True)
    attacks_air = stats.get("attacks_air", False)
 
    # Tank threshold: cards with 1500+ base HP draw significant defensive attention
    is_tank = hitpoints >= 1500
 
    return CardAttributes(
        card_type=card_type,
        damage_type=_parse_damage_type(stats, card_key),
        target_type=_parse_target_type(attacks_ground, attacks_air),
        roles=_assign_roles(card_key, elixir, hitpoints),
        is_win_condition=card_key in WIN_CONDITION_KEYS,
        is_tank=is_tank,
        is_swarm=card_key in SWARM_KEYS,
        is_spell=card_type == CardType.SPELL,
        is_building=card_type == CardType.BUILDING,
        has_air_defense=attacks_air,
    )
 
 
def load_all_cards() -> dict[str, Card]:
    """
    Strategy:
    1. Load cards.json as the master list (120 canonical cards, no evolved variants)
    2. Build a lookup dict from all stats files keyed by 'key' field
    3. For each card in master list, join its stats and build a Card object
 
    The intuition is like a SQL left join — cards.json is the left table,
    stats files are the right table, 'key' is the join column.
    """
    base_cards = [c for c in _load_json("cards.json") if not c.get("is_evolved", False)]
 
    stats_lookup: dict[str, dict] = {}
    for filename in ["cards_stats_characters.json", "cards_stats_spell.json", "cards_stats_building.json"]:
        for entry in _load_json(filename):
            k = entry.get("key")
            if k and not entry.get("is_evolved", False):
                stats_lookup[k] = entry
 
    cards = {}
    for base in base_cards:
        key = base.get("key", "")
        name = base.get("name", "")
        stats = stats_lookup.get(key, {})
 
        # Pull icon URL from the dataset if available
        icon_url = (
            stats.get("iconUrls", {}).get("medium")
            or base.get("iconUrls", {}).get("medium")
            or None
        )
 
        card = Card(
            id=base.get("id", 0),
            name=name,
            elixir_cost=base.get("elixir", 0),
            rarity=_parse_rarity(base.get("rarity", "common")),
            max_level=16,
            attributes=_build_attributes(base, stats),
            icon_url=icon_url
        )
        cards[name.lower()] = card
 
    return cards
 
 
_CARD_REGISTRY: dict[str, Card] | None = None
 
 
def get_card_registry() -> dict[str, Card]:
    global _CARD_REGISTRY
    if _CARD_REGISTRY is None:
        _CARD_REGISTRY = load_all_cards()
    return _CARD_REGISTRY
 