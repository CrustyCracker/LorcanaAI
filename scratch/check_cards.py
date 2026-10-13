from lorcana_ai.data.card_db import GLOBAL_CARD_DB
db = GLOBAL_CARD_DB

cards_to_check = [
    'Ursula - Deceiver',
    'Pegasus - Gift for Hercules',
    'Max Goof - Rebellious Teen',
    'Flynn Rider - Spectral Scoundrel',
    'Vixey - Forest Friend',
    'Tod - All Alone',
    'White Rabbit - Late Again',
    'Elinor - Renowned Diplomat',
    'The Queen - Devious Disguise',
    'Mushu - Stealthy Dragon',
    'Strength of a Raging Fire',
    'Angela - Night Warrior',
    'Angel - Experiment 624',
    'Flash - Efficient Clerk',
    "Woolter & Jesse - Bellwether's Henchmen"
]

for name in cards_to_check:
    c = db.get_by_name(name)
    if c:
        print(f"FOUND: {c.full_name} | {c.ink} | Cost: {c.cost} | Inkable: {c.inkable} | Type: {c.card_type} | Lore: {c.lore}")
    else:
        print(f"NOT FOUND: {name}")
