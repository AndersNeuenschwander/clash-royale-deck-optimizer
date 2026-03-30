from dataclasses import dataclass
from app.core.models import Deck, Card, VulnerabilityProfile
from app.core.analyzer import DeckAnalyzer
from app.core.meta_weighter import ArenaMetaWeights, apply_meta_weights
from app.core.role_registry import RoleConfidence, compute_affinity_bonus
 
 
@dataclass
class SwapSuggestion:
    """
    Represents a single recommended card swap.
 
    The intuition: instead of showing the player a completely new deck,
    we show them incremental improvements — swap THIS card for THAT card
    and here's exactly why (which exposure improves and by how much).
    """
    card_out: Card                   # card to remove from deck
    card_in: Card                    # card to add to deck
    score_before: float              # weighted vulnerability before swap
    score_after: float               # weighted vulnerability after swap
    score_delta: float               # raw vulnerability improvement (positive = better)
    effective_delta: float           # score_delta * (1 + affinity_bonus) — used for ranking
    affinity_bonus: float            # how well card_in fits this deck's winning pattern (0–0.35)
    profile_before: VulnerabilityProfile
    profile_after: VulnerabilityProfile
 
    @property
    def improvement_pct(self) -> float:
        """How much did this swap improve the vulnerability score as a percentage."""
        if self.score_before == 0:
            return 0.0
        return (self.score_delta / self.score_before) * 100
 
 
@dataclass
class OptimizationResult:
    """
    Full result of one optimization pass.
    Contains the best swap found plus runner-up alternatives.
    """
    original_deck: Deck
    original_score: float
    original_profile: VulnerabilityProfile
    best_swap: SwapSuggestion | None
    alternatives: list[SwapSuggestion]   # next best swaps
    weights: ArenaMetaWeights
 
 
class DeckOptimizer:
    """
    Role-aware greedy local search optimizer for Clash Royale decks.
 
    The core idea mirrors how a portfolio manager rebalances:
    rather than rebuilding from scratch, they identify the single
    holding that most improves the risk/return profile and swap it.
    We do the same — find the one card swap that most reduces the
    arena-weighted vulnerability score.
 
    We now add two layers of "game feel" on top of the base algorithm:
 
    1. Role gate (hard constraint):
       A swap is only evaluated if card_in shares at least one CardRole
       with card_out. This prevents nonsense like Knight → Graveyard.
       Knight is {MINI_TANK, CYCLE, DEFENSIVE_SUPPORT}.
       Graveyard is {WIN_CONDITION}.
       Intersection is empty → swap is rejected before scoring even runs.
 
    2. Affinity bonus (soft signal):
       Among valid swaps, we boost the effective score of candidates that
       co-occur with the rest of this deck in the player's winning battles.
       If Ice Golem consistently appears alongside Hog Rider in the player's
       wins, and we're suggesting a MINI_TANK replacement for Knight in a
       Hog Rider deck, Ice Golem gets a bonus over an equally valid but
       less battle-proven alternative.
 
       effective_delta = raw_delta * (1 + affinity_bonus)
 
    Why greedy and not exhaustive?
    With 120 cards and 8 slots, exhaustive search = 120 choose 8 =
    ~800 million combinations. Greedy local search = 8 * 112 = 896
    evaluations per pass. Fast enough to run on every request.
    """
 
    def __init__(self):
        self.analyzer = DeckAnalyzer()
 
    def optimize(
        self,
        deck: Deck,
        card_pool: dict[str, Card],
        weights: ArenaMetaWeights,
        role_confidence: RoleConfidence | None = None,
        top_n: int = 5
    ) -> OptimizationResult:
        """
        Find the best single card swap for a given deck.
 
        Args:
            deck: the player's current 8-card deck
            card_pool: all available cards (from card registry)
            weights: arena meta weights from battle log analysis
            role_confidence: co-occurrence data from winning battles.
                             Pass None to skip the affinity bonus
                             (roles gate still applies).
            top_n: how many alternative swaps to return
 
        Returns:
            OptimizationResult with best swap and alternatives
        """
        # Score the current deck
        original_profile = self.analyzer.analyze(deck)
        original_score = apply_meta_weights(original_profile, weights)
 
        # Cards currently in the deck (to exclude from swap candidates)
        deck_card_names = {c.name.lower() for c in deck.cards}
 
        # Cards NOT in the deck = potential swap-ins
        candidates = [
            card for name, card in card_pool.items()
            if name not in deck_card_names
        ]
 
        # Use an empty RoleConfidence if none was provided —
        # get_deck_affinity returns 0.0 by default, so no bonus is applied
        confidence = role_confidence if role_confidence is not None else RoleConfidence()
 
        # Evaluate every possible swap
        all_swaps: list[SwapSuggestion] = []
 
        for i, card_out in enumerate(deck.cards):
            for card_in in candidates:
 
                # --- Hard gate: role compatibility check ---
                # We intersect the two frozensets of CardRole enums.
                # If there's no overlap, these cards serve completely different
                # functions and we skip this swap entirely — no scoring needed.
                shared_roles = card_out.attributes.roles & card_in.attributes.roles
                if not shared_roles:
                    continue
 
                # Build the swapped deck
                new_cards = list(deck.cards)
                new_cards[i] = card_in
                new_deck = Deck(cards=new_cards)
 
                # Score the new deck's vulnerability
                new_profile = self.analyzer.analyze(new_deck)
                new_score = apply_meta_weights(new_profile, weights)
                raw_delta = original_score - new_score  # positive = improvement
 
                # Only keep swaps that actually reduce vulnerability
                if raw_delta <= 0:
                    continue
 
                # --- Soft signal: affinity bonus from battle data ---
                # The other 7 cards staying in the deck (excluding card_out)
                remaining_deck_cards = [
                    c.name for c in deck.cards if c.name != card_out.name
                ]
                affinity_bonus = compute_affinity_bonus(
                    card_in_name=card_in.name,
                    deck_card_names=remaining_deck_cards,
                    role_confidence=confidence,
                )
 
                # effective_delta is what we rank by:
                # a card that improves vulnerability by 0.05 AND fits the
                # winning deck pattern (bonus=0.30) scores 0.065 effective —
                # beating a card that improves by 0.06 with no battle backing.
                effective_delta = raw_delta * (1.0 + affinity_bonus)
 
                all_swaps.append(SwapSuggestion(
                    card_out=card_out,
                    card_in=card_in,
                    score_before=original_score,
                    score_after=new_score,
                    score_delta=raw_delta,
                    effective_delta=effective_delta,
                    affinity_bonus=affinity_bonus,
                    profile_before=original_profile,
                    profile_after=new_profile,
                ))
 
        # Sort by effective_delta (role-aware + battle-backed) descending
        all_swaps.sort(key=lambda s: s.effective_delta, reverse=True)
 
        # Deduplicate — if the same card_in appears multiple times
        # (swapping different cards out), keep only the highest-ranked version
        seen_card_ins = set()
        deduplicated = []
        for swap in all_swaps:
            if swap.card_in.name not in seen_card_ins:
                seen_card_ins.add(swap.card_in.name)
                deduplicated.append(swap)
 
        best_swap = deduplicated[0] if deduplicated else None
        alternatives = deduplicated[1:top_n] if len(deduplicated) > 1 else []
 
        return OptimizationResult(
            original_deck=deck,
            original_score=original_score,
            original_profile=original_profile,
            best_swap=best_swap,
            alternatives=alternatives,
            weights=weights,
        )
 
    def explain_swap(self, swap: SwapSuggestion) -> str:
        """
        Generate a human-readable explanation of why a swap is recommended.
        This is what will show up on the website as the suggestion text.
 
        We now include role context and affinity information so the player
        understands WHY this card fits — not just what score it improves.
        """
        reasons = []
 
        before = swap.profile_before
        after = swap.profile_after
 
        # Check which vulnerability exposures improved
        if after.air_exposure < before.air_exposure:
            delta = before.air_exposure - after.air_exposure
            reasons.append(f"reduces air exposure by {delta:.0%}")
 
        if after.tank_exposure < before.tank_exposure:
            delta = before.tank_exposure - after.tank_exposure
            reasons.append(f"reduces tank exposure by {delta:.0%}")
 
        if after.swarm_exposure < before.swarm_exposure:
            delta = before.swarm_exposure - after.swarm_exposure
            reasons.append(f"reduces swarm exposure by {delta:.0%}")
 
        if after.spell_exposure < before.spell_exposure:
            delta = before.spell_exposure - after.spell_exposure
            reasons.append(f"reduces spell exposure by {delta:.0%}")
 
        if after.building_exposure < before.building_exposure:
            delta = before.building_exposure - after.building_exposure
            reasons.append(f"reduces building exposure by {delta:.0%}")
 
        if after.elixir_risk < before.elixir_risk:
            delta = before.elixir_risk - after.elixir_risk
            reasons.append(f"reduces elixir risk by {delta:.0%}")
 
        if after.win_condition_score > before.win_condition_score:
            delta = after.win_condition_score - before.win_condition_score
            reasons.append(f"strengthens win condition by {delta:.0%}")
 
        reason_str = ", ".join(reasons) if reasons else "improves overall balance"
 
        # Show shared roles so the player understands why this swap is valid
        shared_roles = swap.card_out.attributes.roles & swap.card_in.attributes.roles
        role_names = ", ".join(r.value.replace("_", " ") for r in shared_roles)
 
        # Show affinity signal if battle data contributed to this suggestion
        affinity_str = ""
        if swap.affinity_bonus > 0.05:
            affinity_str = f", battle-backed ({swap.affinity_bonus:.0%} affinity)"
 
        return (
            f"Swap {swap.card_out.name} → {swap.card_in.name}: "
            f"{reason_str} "
            f"[shared roles: {role_names}{affinity_str}] "
            f"(score: {swap.score_before:.3f} → {swap.score_after:.3f}, "
            f"{swap.improvement_pct:.1f}% improvement)"
        )
 