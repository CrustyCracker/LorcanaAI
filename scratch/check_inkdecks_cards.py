import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding='utf-8')
from lorcana_ai.data.card_db import GLOBAL_CARD_DB

test_names = [
    'Darkwing Duck - Crime Fighter',
    'Broadway - Sturdy and Strong',
    'Launchpad - Trusty Sidekick',
    'The Terror That Flaps in the Night',
    'Darkwing Duck - Shadowy Superhero',
    'He Hurled His Thunderbolt',
    "The Huntsman - On the Queen's Orders",
    'Rapunzel - Tower Defender',
    'Aladdin - Doing His Part',
    'Tod & Copper - Best of Friends',
    'Lenny - Toy Binoculars',
    'You Broke My Smolder',
    'Hamish, Hubert & Harris - Troublemaking Triplets',
    "Look What You've Done",
    'Strike a Good Match',
    'Benja - Guardian of the Dragon Gem',
    'Flynn Rider - Spectral Scoundrel',
    'Rapunzel & Flynn Rider - Unlikely Pair',
    'Gaetan Moliere - Clever Burrower',
    'Bobby Zimuruski - Spray Cheese Kid',
    'Three Arrows',
    'Smash',
    'Fire the Cannons!',
    'Tinker Bell - Giant Fairy',
    'Robin Hood - Champion of Sherwood',
    'Sudden Chill',
]
for n in test_names:
    c = GLOBAL_CARD_DB.get_by_name(n)
    if c:
        ops = [cl.op.name for cl in c.abilities.clauses]
        kw = [k.name for k in [c.abilities.keywords] if k]
        print(f'{n:45s}: cost={c.cost} ink={str(c.inkable):<5} type={c.card_type:10s} stats={c.strength}/{c.willpower}/{c.lore} ops={ops}')
    else:
        print(f'{n:45s}: NOT FOUND')
