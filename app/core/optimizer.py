from dataclasses import dataclass
from app.core.models import Deck, Card, VulnerabilityProfile
from app.core.analyzer import DeckAnalyzer
from app.core.meta_weighter import ArenaMetaWeights, apply_meta_weights
from app.core.role_registry import RoleConfidence, compute_affinity_bonus
from app.ml.cohesion_scorer import DeckCohesionScorer


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
    effective_delta: float           # final ranking score (role + affinity + cohesion)
    affinity_bonus: float            # how well card_in fits winning battle pattern (0–0.35)
    cohesion_score: float            # how well card_in fits deck archetype via ML (0–1)
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


# Minimum cohesion score to pass the ML gate.
# Below this threshold the candidate card is from a sufficiently different
# archetype that we reject it outright, regardless of vulnerability improvement.
# Set conservatively at 0.15 so we only block clear mismatches (not edge cases).
# When embeddings aren't loaded the scorer returns 0.5, so this gate is inactive.
MIN_COHESION = 0.15

# Maps raw cohesion score [0, 1] → effective weight [COHESION_FLOOR, 1.0].
# This prevents cohesion from fully killing a large vulnerability improvement —
# a card with cohesion=0 still contributes COHESION_FLOOR of its raw_delta.
# A card with cohesion=1.0 contributes its full raw_delta (multiplier = 1.0).
COHESION_FLOOR = 0.70


def _cohesion_multiplier(cohesion_score: float) -> float:
    """
    Convert a cohesion score [0, 1] to an effective delta multiplier [COHESION_FLOOR, 1.0].

    cohesion = 1.0  →  multiplier = 1.0   (perfect archetype fit, no penalty)
    cohesion = 0.5  →  multiplier = 0.85  (neutral / no embedding data)
    cohesion = 0.0  →  multiplier = 0.70  (different archetype, 30% penalty)

    This is a soft signal — it penalizes bad fits but never zeroes out a swap
    that has a large vulnerability improvement. The hard gate (MIN_COHESION)
    handles the truly egregious mismatches.
    """
    return COHESION_FLOOR + (1.0 - COHESION_FLOOR) * cohesion_score


class DeckOptimizer:
    """
    Role-aware, ML-backed greedy local search optimizer for Clash Royale decks.

    The core idea mirrors how a portfolio manager rebalances:
    rather than rebuilding from scratch, identify the single holding that most
    improves the risk/return profile and swap it.

    Three layers of "game feel" stack on top of the base vulnerability score:

    1. Role gate (hard constraint):
       A swap is only evaluated if card_in shares at least one CardRole
       with card_out. This prevents gross mismatches: Knight is
       {MINI_TANK, CYCLE, DEFENSIVE_SUPPORT}; Graveyard is {WIN_CONDITION}.
       Intersection is empty → swap is rejected before scoring even runs.

    2. ML cohesion gate + soft signal (Card2Vec):
       Even if two cards share a role, they might be from completely different
       deck archetypes. Card2Vec embeddings (trained on thousands of real decks)
       capture which cards belong together. We apply two cohesion checks:

       a. Hard gate: if cohesion_score < MIN_COHESION (0.15), the swap is
          rejected. This catches clear archetype violations the role gate misses
          (e.g. a card with multiple roles that technically overlaps but is
          semantically wrong for this deck).

       b. Soft multiplier: cohesion_score maps to [0.70, 1.0] and is applied
          to effective_delta, penalizing weak-fit cards without zeroing them out.

    3. Affinity bonus (battle-derived soft signal):
       Among remaining valid swaps, cards that co-occurred with the current
       deck's cards in the player's own winning battles get a bonus multiplier.
       This personalizes the ranking to the individual player's style.

    Final ranking:
        effective_delta = raw_delta * cohesion_multiplier(cohesion) * (1 + affinity_bonus)

    Why greedy and not exhaustive?
    With 120 cards and 8 slots, exhaustive search = ~800 million combinations.
    Greedy local search = 8 × 112 = 896 evaluations per pass. Sub-second.
    """

    def __init__(self, cohesion_scorer: DeckCohesionScorer | None = None):
        """
        Args:
            cohesion_scorer: pre-loaded DeckCohesionScorer instance.
                             If None, a new scorer is created (loads embeddings
                             from disk if available). Pass DeckCohesionScorer({})
                             to explicitly disable ML scoring.
        """
        self.analyzer = DeckAnalyzer()
        self.cohesion_scorer = (
            cohesion_scorer if cohesion_scorer is not None
            else DeckCohesionScorer()
        )

    def optimize(
        self,
        deck: Deck,
        card_pool: dict[str, Card],
        weights: ArenaMetaWeights,
        role_confidence: RoleConfidence | None = None,
        top_n: int = 5,
    ) -> OptimizationResult:
        """
        Find the best single card swap for a given deck.

        Args:
            deck: the player's current 8-card deck
            card_pool: all available cards (from card registry)
            weights: arena meta weights from battle log analysis
            role_confidence: co-occurrence data from winning battles.
                             Pass None to skip the affinity bonus
                             (role gate and cohesion gate still apply).
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

        all_swaps: list[SwapSuggestion] = []

        for i, card_out in enumerate(deck.cards):
            for card_in in candidates:

                # --- Gate 1: Role compatibility (hard) ---
                # Intersect the two frozensets of CardRole enums.
                # No overlap = cards serve completely different functions.
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

                # --- Gate 2: ML cohesion (hard + soft) ---
                # The remaining 7 cards (excluding card_out) form the context.
                remaining_deck_cards = [
                    c.name for c in deck.cards if c.name != card_out.name
                ]

                cohesion = self.cohesion_scorer.score(
                    card_in_name=card_in.name,
                    remaining_card_names=remaining_deck_cards,
                )

                # Hard gate: reject clear archetype violations.
                # When embeddings aren't loaded, cohesion = 0.5 → gate is inactive.
                if cohesion < MIN_COHESION:
                    continue

                # Soft multiplier: penalizes weak archetype fits without zeroing them.
                # cohesion=1.0 → multiplier=1.0 (no penalty)
                # cohesion=0.5 → multiplier=0.85 (slight penalty)
                # cohesion=0.15 → multiplier=0.705 (just above floor, just passed gate)
                c_mult = _cohesion_multiplier(cohesion)

                # --- Signal 3: Affinity bonus (soft, battle-derived) ---
                affinity_bonus = compute_affinity_bonus(
                    card_in_name=card_in.name,
                    deck_card_names=remaining_deck_cards,
                    role_confidence=confidence,
                )

                # Final ranking score:
                #   raw_delta × cohesion_multiplier × (1 + affinity_bonus)
                #
                # Cohesion (ML archetype fit) and affinity (personal battle data)
                # both act as multipliers on the vulnerability improvement.
                # A card that improves vulnerability by 0.05, fits the archetype
                # perfectly (cohesion=1.0), and has battle backing (affinity=0.30)
                # scores 0.05 × 1.0 × 1.30 = 0.065 effective.
                # A card that improves by 0.08 but is a bad archetype fit
                # (cohesion=0.2, mult=0.76) and no battle backing scores
                # 0.08 × 0.76 × 1.0 = 0.061 effective — loses to the better fit.
                effective_delta = raw_delta * c_mult * (1.0 + affinity_bonus)

                all_swaps.append(SwapSuggestion(
                    card_out=card_out,
                    card_in=card_in,
                    score_before=original_score,
                    score_after=new_score,
                    score_delta=raw_delta,
                    effective_delta=effective_delta,
                    affinity_bonus=affinity_bonus,
                    cohesion_score=cohesion,
                    profile_before=original_profile,
                    profile_after=new_profile,
                ))

        # Sort by effective_delta (role + ML cohesion + battle affinity) descending
        all_swaps.sort(key=lambda s: s.effective_delta, reverse=True)

        # Deduplicate — if the same card_in appears multiple times
        # (swapping different cards out), keep only the highest-ranked version
        seen_card_ins: set[str] = set()
        deduplicated: list[SwapSuggestion] = []
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

        Now includes cohesion context so the player understands the ML signal —
        not just what score improved, but how well the card fits the deck's style.
        """
        reasons = []

        before = swap.profile_before
        after = swap.profile_after

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

        # Show cohesion signal if ML embeddings are active
        cohesion_str = ""
        if swap.cohesion_score != 0.5:  # 0.5 = neutral / no embeddings loaded
            cohesion_pct = int(swap.cohesion_score * 100)
            cohesion_str = f", archetype fit {cohesion_pct}%"

        # Show affinity signal if battle data contributed to this suggestion
        affinity_str = ""
        if swap.affinity_bonus > 0.05:
            affinity_str = f", battle-backed ({swap.affinity_bonus:.0%} affinity)"

        return (
            f"Swap {swap.card_out.name} → {swap.card_in.name}: "
            f"{reason_str} "
            f"[shared roles: {role_names}{cohesion_str}{affinity_str}] "
            f"(score: {swap.score_before:.3f} → {swap.score_after:.3f}, "
            f"{swap.improvement_pct:.1f}% improvement)"
        )
