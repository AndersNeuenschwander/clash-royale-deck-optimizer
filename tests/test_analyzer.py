import app.data.card_loader as cl
cl._CARD_REGISTRY = None
from app.core.models import Deck
from app.core.analyzer import DeckAnalyzer

cards = cl.get_card_registry()

hog_cycle = Deck(cards=[
    cards['hog rider'],
    cards['musketeer'],
    cards['ice golem'],
    cards['fireball'],
    cards['the log'],
    cards['skeletons'],
    cards['ice spirit'],
    cards['cannon'],
])

analyzer = DeckAnalyzer()
profile = analyzer.analyze(hog_cycle)

print(f'Avg elixir: {hog_cycle.average_elixir:.2f}')
print(f'Air exposure:      {profile.air_exposure:.2f}')
print(f'Tank exposure:     {profile.tank_exposure:.2f}')
print(f'Swarm exposure:    {profile.swarm_exposure:.2f}')
print(f'Spell exposure:    {profile.spell_exposure:.2f}')
print(f'Building exposure: {profile.building_exposure:.2f}')
print(f'Elixir risk:       {profile.elixir_risk:.2f}')
print(f'Win con score:     {profile.win_condition_score:.2f}')
print(f'Overall score:     {profile.overall_score:.2f}')