from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
 
 
@dataclass
class RoleConfidence:
    """
    Stores co-occurrence confidence scores derived from battle logs.
 
    The core idea: if Card A and Card B appear together in winning decks
    frequently, they probably serve complementary roles and a replacement
    for A should ideally also pair well with B.
 
    We represent this as a nested dict:
        confidence[card_a][card_b] = float between 0.0 and 1.0
 
    Where 1.0 means "every winning deck that contained card_a also
    contained card_b" — they are almost always played together.
 
    Think of this like a correlation matrix in portfolio theory:
    assets (cards) that are highly correlated tend to serve the same
    market condition (game situation). When you swap one out, you want
    to bring in something with a similar correlation profile.
    """
    # confidence[card_name][partner_name] = co-occurrence rate in winning decks
    confidence: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    # How many winning battles the confidence scores are based on
    sample_size: int = 0
 
    def get_partner_confidence(self, card_name: str, partner_name: str) -> float:
        """
        How often does `partner_name` appear in winning decks that
        also contain `card_name`?
 
        Returns 0.0 if we have no data (safe default — no bonus, no penalty).
        """
        return self.confidence.get(card_name.lower(), {}).get(partner_name.lower(), 0.0)
 
    def get_deck_affinity(
        self,
        candidate_card_name: str,
        current_deck_card_names: list[str]
    ) -> float:
        """
        How well does a candidate card fit the rest of the current deck,
        based on how often it co-occurred with those cards in winning battles?
 
        The intuition: we're asking "does this card belong in a deck like
        this one?" We average its pairwise confidence with every card
        already in the deck. A card that consistently appears alongside
        Hog Rider, Musketeer, and Ice Golem in winning decks will score
        high affinity for a Hog Rider deck — even if its raw vulnerability
        improvement is modest.
 
        This is the "game feel" signal: it rewards cards that players
        have already proven work well together.
        """
        if not current_deck_card_names or self.sample_size == 0:
            return 0.0
 
        candidate = candidate_card_name.lower()
        scores = []
        for deck_card in current_deck_card_names:
            score = self.confidence.get(candidate, {}).get(deck_card.lower(), 0.0)
            scores.append(score)
 
        return sum(scores) / len(scores) if scores else 0.0
 
 
def build_role_confidence(battles: list[dict], player_tag: str) -> RoleConfidence:
    """
    Analyze the player's battle log to learn which cards appear together
    in WINNING decks at their trophy range.
 
    We only look at the player's OWN winning decks (not opponents), because
    we want to learn what works FOR this player in their arena. The opponent
    data is already used by meta_weighter.py to understand threats; this
    function is about understanding successful deck compositions.
 
    Algorithm:
    1. Filter to battles the player WON
    2. Extract the 8-card deck the player used in each win
    3. For each pair of cards (A, B) in a winning deck, increment
       co_occurrence[A][B] and co_occurrence[B][A]
    4. Normalize by how many winning decks each card appeared in
       to get a rate rather than a raw count
 
    The result is: "given that Card A is in my deck and I'm winning,
    Card B appears alongside it X% of the time."
 
    Args:
        battles: raw battle log dicts from the Clash Royale API
        player_tag: the player's tag (used to identify which side is 'us')
 
    Returns:
        RoleConfidence with populated co-occurrence scores
    """
    if not battles:
        return RoleConfidence()
 
    # Normalize tag format — API returns '#ABC' but player might store 'ABC'
    normalized_tag = player_tag.strip().lstrip("#").upper()
 
    # co_occurrence[card_a][card_b] = number of winning decks containing both
    co_occurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
 
    # appearance_in_wins[card] = number of winning decks this card appeared in
    appearance_in_wins: dict[str, int] = defaultdict(int)
 
    winning_battles = 0
 
    for battle in battles:
        # Identify which side is our player
        team = battle.get("team", [])
        if not team:
            continue
 
        # The API puts the player's data in 'team', opponent in 'opponent'
        # We check crowns to determine if the player won
        player_side = team[0]
        opponent_side = battle.get("opponent", [{}])[0]
 
        player_crowns = player_side.get("crowns", 0)
        opponent_crowns = opponent_side.get("crowns", 0)
 
        # Only process winning battles — we want to learn from success
        if player_crowns <= opponent_crowns:
            continue
 
        player_cards = player_side.get("cards", [])
        if not player_cards:
            continue
 
        # Extract card names (lowercased for consistent keying)
        card_names = [c["name"].lower() for c in player_cards]
 
        # Count every card's appearance in this win
        for card in card_names:
            appearance_in_wins[card] += 1
 
        # Count every pair co-occurrence in this winning deck
        # We do both directions (A→B and B→A) so lookups work either way
        for i, card_a in enumerate(card_names):
            for card_b in card_names:
                if card_a != card_b:
                    co_occurrence[card_a][card_b] += 1
 
        winning_battles += 1
 
    if winning_battles == 0:
        return RoleConfidence(sample_size=0)
 
    # Normalize: convert raw counts to rates
    # confidence[A][B] = (times A and B appeared together in wins)
    #                    / (times A appeared in wins)
    #
    # This gives us a conditional probability:
    # "Given that A is in my winning deck, how often is B also there?"
    #
    # A value of 0.8 means: 80% of wins that included A also included B.
    # That's a strong signal that B belongs in the same deck as A.
    confidence: dict[str, dict[str, float]] = {}
 
    for card_a, partners in co_occurrence.items():
        total_wins_with_a = appearance_in_wins[card_a]
        if total_wins_with_a == 0:
            continue
        confidence[card_a] = {
            card_b: count / total_wins_with_a
            for card_b, count in partners.items()
        }
 
    return RoleConfidence(
        confidence=confidence,
        sample_size=winning_battles,
    )
 
 
def compute_affinity_bonus(
    card_in_name: str,
    deck_card_names: list[str],
    role_confidence: RoleConfidence,
    max_bonus: float = 0.35,
) -> float:
    """
    Compute the affinity bonus to apply to a swap's score delta.
 
    This is the bridge between battle data and the optimizer's scoring.
    A swap that improves vulnerability AND fits the winning deck pattern
    gets a bonus multiplier. A swap that improves vulnerability but the
    candidate card never appears in winning decks with the current cards
    gets no bonus (but is still a valid swap — the hard gate is roles).
 
    The bonus is additive to the score_delta, scaled by affinity:
        effective_delta = raw_delta * (1 + affinity_bonus)
 
    We cap the bonus at max_bonus (default 35%) so battle data can
    *influence* rankings but never *override* a large vulnerability
    improvement with a tiny one just because of co-occurrence.
 
    Args:
        card_in_name: the candidate card being swapped in
        deck_card_names: names of all other cards staying in the deck
        role_confidence: the RoleConfidence object from build_role_confidence
        max_bonus: ceiling on the multiplier bonus (default 0.35 = 35%)
 
    Returns:
        A float between 0.0 and max_bonus representing the bonus multiplier
    """
    affinity = role_confidence.get_deck_affinity(card_in_name, deck_card_names)
    # Scale linearly: affinity of 1.0 → full max_bonus, 0.0 → no bonus
    return affinity * max_bonus
 