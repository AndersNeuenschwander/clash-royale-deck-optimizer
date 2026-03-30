from app.core.models import (
    Deck, VulnerabilityProfile, DamageType, TargetType, CardType
)


class DeckAnalyzer:
    """
    Computes a VulnerabilityProfile for a given deck.

    The intuition mirrors factor analysis in quant finance:
    each 'exposure' is a factor score measuring how vulnerable
    the deck is to a specific threat archetype.
    All scores are normalized to [0, 1] where 1 = maximally vulnerable.
    """

    def analyze(self, deck: Deck) -> VulnerabilityProfile:
        return VulnerabilityProfile(
            deck=deck,
            air_exposure=self._air_exposure(deck),
            tank_exposure=self._tank_exposure(deck),
            swarm_exposure=self._swarm_exposure(deck),
            spell_exposure=self._spell_exposure(deck),
            building_exposure=self._building_exposure(deck),
            elixir_risk=self._elixir_risk(deck),
            win_condition_score=self._win_condition_score(deck),
        )

    def _air_exposure(self, deck: Deck) -> float:
        """
        How vulnerable is this deck to air units (Balloon, Minions, Lava Hound)?

        We count cards that can reliably handle air — either they explicitly
        target air, or they deal splash damage (which hits air units too).
        The fewer air counters, the higher the exposure.
        """
        air_counters = sum(
            1 for c in deck.cards
            if c.attributes.target_type in (TargetType.AIR, TargetType.BOTH)
            or (
                c.attributes.damage_type == DamageType.SPLASH
                and c.attributes.target_type == TargetType.BOTH
            )
        )
        # 0 counters = fully exposed, 3+ counters = well covered
        return max(0.0, 1.0 - (air_counters / 3.0))

    def _tank_exposure(self, deck: Deck) -> float:
        """
        How vulnerable is this deck to high-HP tanks (Giant, Golem, PEKKA)?

        A deck needs either: high single-target DPS cards, buildings that
        distract tanks, or spells that chip them down.
        We proxy this by checking for spells and high-damage single-target cards.
        """
        tank_answers = sum(
            1 for c in deck.cards
            if c.attributes.is_spell                          # spells like Rocket, Lightning
            or c.attributes.is_building                       # buildings distract tanks
            or (
                c.attributes.damage_type == DamageType.SINGLE
                and c.elixir_cost >= 4                        # high-cost = high DPS single target
            )
        )
        return max(0.0, 1.0 - (tank_answers / 3.0))

    def _swarm_exposure(self, deck: Deck) -> float:
        """
        How vulnerable is this deck to swarm cards (Goblin Gang, Skeleton Army)?

        Swarms are countered by splash damage spells or splash troops.
        A deck with no splash is very exposed to swarms.
        """
        splash_cards = sum(
            1 for c in deck.cards
            if c.attributes.damage_type == DamageType.SPLASH
        )
        return max(0.0, 1.0 - (splash_cards / 2.0))

    def _spell_exposure(self, deck: Deck) -> float:
        """
        How vulnerable is this deck to spells (Fireball, Lightning, Arrows)?

        Decks that rely on clumped swarm cards or multiple low-HP units
        placed together are highly exposed to area spells.
        We measure this by counting swarm cards and low-elixir multi-unit cards.
        """
        spell_bait_cards = sum(
            1 for c in deck.cards
            if c.attributes.is_swarm
            or (c.elixir_cost <= 2 and not c.attributes.is_spell)
        )
        # 3+ spell-bait cards in a deck means you're very exposed to spells
        return min(1.0, spell_bait_cards / 3.0)

    def _building_exposure(self, deck: Deck) -> float:
        """
        How vulnerable is this deck to defensive buildings (Inferno Tower, Tesla)?

        Win condition decks that can't remove buildings get hard-countered.
        We check if the deck has a win condition AND lacks spell coverage.
        """
        has_win_con = deck.has_win_condition
        spell_count = len(deck.spells)

        if not has_win_con:
            return 0.0  # no win con = buildings aren't a specific threat

        # Win con deck with no spells = fully exposed to buildings
        if spell_count == 0:
            return 1.0
        elif spell_count == 1:
            return 0.4
        else:
            return 0.1

    def _elixir_risk(self, deck: Deck) -> float:
        """
        How risky is the deck's elixir curve?

        This captures two failure modes:
        1. Too expensive (avg > 4.5) — you can't cycle fast enough, opponent
           takes your tower while you wait for elixir
        2. No cheap cycle cards — you get stuck waiting for your win condition

        The intuition: in Clash Royale, tempo is everything. An expensive deck
        that can't defend early is like a high-beta portfolio with no hedge.
        """
        avg = deck.average_elixir
        cheap_cards = sum(1 for c in deck.cards if c.is_cheap)

        # Score the avg elixir risk (normalized: 3.0 is fine, 5.0 is very risky)
        elixir_curve_risk = max(0.0, min(1.0, (avg - 3.0) / 2.5))

        # Penalize decks with fewer than 2 cheap cycle cards
        cycle_risk = max(0.0, 1.0 - (cheap_cards / 2.0))

        # Weighted combination — elixir curve matters more than cycle risk
        return 0.6 * elixir_curve_risk + 0.4 * cycle_risk

    def _win_condition_score(self, deck: Deck) -> float:
        """
        How strong is the deck's offensive win condition? (higher = better)

        Unlike the other scores where higher = more vulnerable,
        this is an offensive strength metric. We factor it into the
        overall score as a negative (it offsets vulnerability).

        A deck with no win condition scores 0 — it has no clear path to victory.
        Multiple win conditions score higher but with diminishing returns.
        """
        win_cons = deck.win_conditions
        if not win_cons:
            return 0.0

        # Primary win condition value — expensive win cons hit harder
        primary = win_cons[0]
        base_score = min(1.0, primary.elixir_cost / 6.0)

        # Bonus for having a secondary win condition (diminishing returns)
        if len(win_cons) > 1:
            base_score = min(1.0, base_score + 0.2)

        return base_score