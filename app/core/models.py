from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
 
 
# --- Enums define the "vocabulary" of our domain ---
# Using enums instead of raw strings means typos become errors immediately.
 
class CardType(Enum):
    TROOP = "troop"
    SPELL = "spell"
    BUILDING = "building"
 
class Rarity(Enum):
    COMMON = "common"
    RARE = "rare"
    EPIC = "epic"
    LEGENDARY = "legendary"
    CHAMPION = "champion"
 
class DamageType(Enum):
    SINGLE = "single"       # targets one unit at a time
    SPLASH = "splash"       # damages all units in an area
    CHAIN = "chain"         # jumps between targets (e.g. Electro Wizard)
 
class TargetType(Enum):
    GROUND = "ground"
    AIR = "air"
    BOTH = "both"
    BUILDINGS = "buildings"
 
 
class CardRole(Enum):
    """
    The functional role a card plays inside a deck.
 
    This is the core of "game feel" — the optimizer uses roles as a hard
    constraint so it only suggests swaps between cards that serve the same
    purpose. A card can have multiple roles (e.g. Valkyrie is both a
    MINI_TANK and a SWARM_CLEAR), and a swap is valid as long as the
    incoming card covers at least one role the outgoing card held.
 
    Think of roles like positions on a sports team: you can swap one
    midfielder for another, but you can't replace a goalkeeper with a
    striker just because both are players.
    """
 
    # --- Offensive roles ---
    WIN_CONDITION  = "win_condition"
    # The card whose job is to threaten or damage the enemy tower.
    # Examples: Hog Rider, Balloon, Graveyard, Miner, X-Bow.
    # These are irreplaceable — you never swap a win condition for a support card.
 
    SECONDARY_WIN_CONDITION = "secondary_win_condition"
    # A second offensive threat that creates a two-pronged attack.
    # Examples: Miner (paired with Poison), Battle Ram, Wall Breakers.
    # Less committing than a primary win condition but still offensive.
 
    SPELL_SUPPORT  = "spell_support"
    # A spell whose job is to enable or protect the win condition push.
    # Examples: Fireball (clears support troops), Poison (slows defense),
    # Zap/Log (resets Inferno, kills swarms).
 
    # --- Defensive / utility roles ---
    MINI_TANK      = "mini_tank"
    # A moderately high-HP card that walks in front of the win condition,
    # absorbing tower shots and defensive troops.
    # Examples: Knight, Ice Golem, Dark Prince, Battle Ram.
    # Key property: cheap enough to cycle, tanky enough to take hits.
 
    TANK           = "tank"
    # A high-HP card that anchors a slow, heavy push.
    # Examples: Giant, Golem, Lava Hound, PEKKA.
    # These are typically win conditions themselves in beatdown decks.
 
    SUPPORT_TROOP  = "support_troop"
    # A ranged or medium card that follows behind a tank, dealing damage
    # to defending troops while the tank soaks hits.
    # Examples: Musketeer, Witch, Executioner, Baby Dragon, Mega Minion.
 
    DEFENSIVE_SUPPORT = "defensive_support"
    # A card primarily used to defend against incoming pushes.
    # Examples: Knight (blocks tanks), Valkyrie (clears swarms),
    # Mega Knight (punishes swarms and tanks), Ice Wizard (slows pushes).
 
    SWARM_CLEAR    = "swarm_clear"
    # A card whose defining job is destroying groups of small units
    # (Skeleton Army, Goblin Gang, Minion Horde).
    # Examples: Valkyrie, Witch, Executioner, Arrows, Log, Fireball.
 
    ANTI_AIR       = "anti_air"
    # A card that can reliably shoot down air units.
    # Examples: Mega Minion, Minions, Archers, Inferno Dragon, Tesla.
    # Without one of these, Balloon and Lava Hound decks win for free.
 
    TANK_KILLER    = "tank_killer"
    # A card with high single-target DPS specifically for destroying tanks.
    # Examples: Inferno Tower, Inferno Dragon, PEKKA, Mini PEKKA, Tesla.
    # The defining property: damage scales up over time on one target.
 
    CYCLE          = "cycle"
    # A cheap (1-3 elixir) card whose main job is to cycle the rotation
    # back to the win condition faster.
    # Examples: Skeletons, Ice Spirit, Goblin, Bats, Zap.
    # Swapping a cycle card for an expensive card breaks the entire deck rhythm.
 
    BUILDING       = "building"
    # A stationary structure — either defensive (Tesla, Cannon) or
    # offensive (X-Bow, Mortar). Buildings occupy a dedicated deck slot
    # and cannot be replaced by troops or spells.
 
    SPAWNER        = "spawner"
    # A building or card that continuously produces units over time.
    # Examples: Goblin Hut, Barbarian Hut, Tombstone, Night Witch.
    # Provides sustained pressure rather than a single burst.
 
    PUNISHMENT     = "punishment"
    # A card played specifically to punish an opponent who overcommits
    # elixir on offense, by launching a devastating counter-push.
    # Examples: Skeleton Army (stops a lone tank cold), Giant Skeleton,
    # P.E.K.K.A (placed behind a surviving tank becomes a push).
 
 
# --- CardAttributes is our hand-defined structural layer ---
# The API doesn't give us these — we define them ourselves per card.
@dataclass
class CardAttributes:
    card_type: CardType
    damage_type: DamageType
    target_type: TargetType
    roles: frozenset                 # frozenset[CardRole] — the card's functional roles
    is_win_condition: bool = False   # e.g. Hog Rider, Giant, Balloon
    is_tank: bool = False            # high HP, draws defensive attention
    is_swarm: bool = False           # spawns multiple low-HP units
    is_spell: bool = False           # no unit, just effect
    is_building: bool = False        # stationary defensive/offensive structure
    has_air_defense: bool = False    # can it shoot flying units?
 
 
# --- Card is the core domain object ---
# It combines what the API tells us (elixir, rarity, id)
# with our hand-defined structural attributes.
@dataclass
class Card:
    id: int
    name: str
    elixir_cost: int
    rarity: Rarity
    max_level: int
    attributes: CardAttributes
    icon_url: Optional[str] = None
 
    @property
    def is_cheap(self) -> bool:
        """Cards costing 3 or less elixir are considered cycle cards."""
        return self.elixir_cost <= 3
 
    @property
    def is_expensive(self) -> bool:
        """Cards costing 6+ elixir are heavy hitters — high risk/reward."""
        return self.elixir_cost >= 6
 
 
# --- Deck is a collection of exactly 8 cards ---
# This is the object we'll run all our analysis on.
@dataclass
class Deck:
    cards: list[Card] = field(default_factory=list)
 
    def __post_init__(self):
        if len(self.cards) > 8:
            raise ValueError("A deck cannot have more than 8 cards")
 
    @property
    def average_elixir(self) -> float:
        """
        Average elixir cost — the single most important deck metric.
        Lower = faster cycle, higher = more powerful individual cards.
        The tradeoff is at the heart of Clash Royale strategy.
        """
        if not self.cards:
            return 0.0
        return sum(c.elixir_cost for c in self.cards) / len(self.cards)
 
    @property
    def has_win_condition(self) -> bool:
        return any(c.attributes.is_win_condition for c in self.cards)
 
    @property
    def win_conditions(self) -> list[Card]:
        return [c for c in self.cards if c.attributes.is_win_condition]
 
    @property
    def spells(self) -> list[Card]:
        return [c for c in self.cards if c.attributes.is_spell]
 
    @property
    def buildings(self) -> list[Card]:
        return [c for c in self.cards if c.attributes.is_building]
 
    @property
    def can_handle_air(self) -> bool:
        """Does any card in the deck target air units?"""
        return any(
            c.attributes.target_type in (TargetType.AIR, TargetType.BOTH)
            or c.attributes.has_air_defense
            for c in self.cards
        )
 
 
# --- VulnerabilityProfile is the output of our analyzer ---
# Think of this as the "risk report" for a deck —
# analogous to a portfolio's factor exposure report in quant finance.
@dataclass
class VulnerabilityProfile:
    deck: Deck
    air_exposure: float = 0.0        # how vulnerable to air attacks (0-1)
    tank_exposure: float = 0.0       # how vulnerable to high-HP tanks (0-1)
    swarm_exposure: float = 0.0      # how vulnerable to swarm cards (0-1)
    spell_exposure: float = 0.0      # how vulnerable to splash spells (0-1)
    building_exposure: float = 0.0   # no counters to defensive buildings (0-1)
    elixir_risk: float = 0.0         # deck too expensive / no cycle cards (0-1)
    win_condition_score: float = 0.0 # strength of offensive win condition (0-1)
 
    @property
    def overall_score(self) -> float:
        """
        Composite vulnerability score — higher is worse.
        Later we'll weight each factor by arena meta usage rates.
        For now, equal weights as a starting baseline.
        """
        exposures = [
            self.air_exposure,
            self.tank_exposure,
            self.swarm_exposure,
            self.spell_exposure,
            self.building_exposure,
            self.elixir_risk,
        ]
        return sum(exposures) / len(exposures)
 