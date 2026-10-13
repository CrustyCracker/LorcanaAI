#!/usr/bin/env python3
"""
Optymalizator kart techowych i pakietów flex w mirrorze Green/Steel (Emerald/Steel)
oparty na realnych danych turniejowych z Inkdecks.com:
https://inkdecks.com/lorcana-metagame/core/emerald-steel/stats

Zamiast losowych/nieużywanych kart ze starych setów (np. Tinker Bell - Tiny Tactician),
skrypt testuje dokładnie te karty techowe i pakiety, którymi grają najlepsi gracze
w turniejach Core Constructed (Darkwing Duck, Launchpad, Broadway, The Terror That Flaps,
He Hurled His Thunderbolt, The Huntsman, Lenny Toy Binoculars, Tod & Copper itp.).

Gracz 2: Zablokowana lista bazowa 60 kart.
Gracz 1: Zablokowany 40-kartowy rdzeń + 20 slotów flex/tech.
Domyślny seed: 2137.
Domyślny lookahead_depth: 2.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lorcana_ai.config import LOG_DIR, RESULTS_DIR
from lorcana_ai.data.card_db import GLOBAL_CARD_DB, CardTemplate
from lorcana_ai.data.deck_io import Deck
from lorcana_ai.data.inkdecks_scraper import scrape_inkdecks_emerald_steel
from lorcana_ai.data.meta_decks import (
    get_green_steel_core_counts,
    get_green_steel_baseline_deck,
)
from lorcana_ai.deckbuilder.matchup_evaluator import MatchupEvaluator
from lorcana_ai.deckbuilder.mirror_optimizer import (
    GreenSteelMirrorOptimizer,
    MirrorTechEvaluation,
    MirrorFlexPackage,
)
from lorcana_ai.engine.game_replay import simulate_and_format_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Testowanie realnych techów z Inkdecks.com w mirrorze Green/Steel (Emerald/Steel)"
    )
    parser.add_argument(
        "--games-tech",
        type=int,
        default=50,
        help="Liczba gier na każdą kartę techową z Inkdecks w Fazie 1 (domyślnie: 50)",
    )
    parser.add_argument(
        "--games-packages",
        type=int,
        default=80,
        help="Liczba gier na każdy 20-kartowy pakiet flex w Fazie 2 (domyślnie: 80)",
    )
    parser.add_argument(
        "--lookahead-depth",
        type=int,
        default=2,
        help="Głębokość analizy taktycznej agenta (domyślnie: 2 - pełny crackback)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Liczba procesów roboczych CPU (0 = auto-detekcja)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2137,
        help="Ziarno losowości dla Common Random Numbers (domyślnie: 2137)",
    )
    parser.add_argument(
        "--refresh-inkdecks",
        action="store_true",
        help="Wymuszenie ponownego pobrania najświeższych statystyk z Inkdecks.com",
    )
    return parser.parse_args()


def explain_inkdecks_mirror_role(card: CardTemplate, presence_pct: float, avg_copies: float) -> str:
    """Szczegółowe wyjaśnienie roli karty z Inkdecks w mirrorze Green/Steel."""
    name = card.full_name
    meta_stat = f"[Inkdecks: {presence_pct:.1f}% obecności w taliach, śr. {avg_copies:.1f}x] "

    if "Darkwing Duck - Crime Fighter" in name:
        return (
            meta_stat
            + "1-drop 2/2 (inkowalny). Fundament nowoczesnego Green Steela: wymienia się natychmiast "
            "z wrogim Tod - All Alone (2/2) oraz Flynn Riderem (1/2), odbierając rywalowi przewagę tempa."
        )
    if "Broadway - Sturdy and Strong" in name:
        return (
            meta_stat
            + "5-drop 4/6 za 5 ink (inkowalny). Potężne ciało nie do zabicia dla 'Strength of a Raging Fire' "
            "we wczesnej fazie. Dominuje naziemną walkę i blokuje każdego wrogiego questera."
        )
    if "Launchpad - Trusty Sidekick" in name:
        return (
            meta_stat
            + "2-drop 2/3 (uninkable). Kluczowy silnik dobierania kart: za każdym razem, gdy questuje lub zostaje wytapowany, "
            "dobiera kartę! Wygrywa wojnę na zasoby w mirrorze, gdzie gracze szybko wchodzą w topdeck."
        )
    if "The Terror That Flaps in the Night" in name:
        return (
            meta_stat
            + "Akcja za 2 ink zadająca 2 obrażenia. Błyskawiczny removal niszczący wrogiego Flynn Ridera, "
            "Ursulę czy White Rabbita, zanim zdążą zapunktować."
        )
    if "Darkwing Duck - Shadowy Superhero" in name:
        return (
            meta_stat
            + "3-drop 2/5 (uninkable). Na wejściu zadaje 2 obrażenia wybranej postaci, a z 5 punktami życia "
            "przeżywa ataki i wymusza niekorzystne wymiany na przeciwniku."
        )
    if "He Hurled His Thunderbolt" in name:
        return (
            meta_stat
            + "Akcja za 4 ink zadająca aż 4 obrażenia. Bezwzględny twardy removal likwidujący wysokie zagrożenia "
            "takie jak The Queen, Elinor czy wrogie Evasive postacie bez utraty własnych ciał."
        )
    if "The Huntsman - On the Queen's Orders" in name:
        return (
            meta_stat
            + "3-drop 4/3 za 3 ink (inkowalny). Ogromna siła uderzeniowa 4 ¤ pozwala jednym ciosem zabić "
            "wrogiego Angela, Mushu czy Flasha."
        )
    if "Rapunzel - Tower Defender" in name:
        return (
            meta_stat
            + "4-drop 3/3 (inkowalny). Na wejściu pozwala odrzucić kartę, by cofnąć na rękę kluczową postać rywala "
            "(np. The Queen, Elinor, Tod & Copper), niszcząc zainwestowane przez rywala tempo."
        )
    if "Tod & Copper - Best of Friends" in name:
        return (
            meta_stat
            + "4-drop 3/3 questujący za 2 Lore z EVASIVE (inkowalny). Bezpiecznie zbiera punkty Lore ponad głowami "
            "naziemnych obrońców przeciwnika."
        )
    if "Lenny - Toy Binoculars" in name:
        return (
            meta_stat
            + "3-drop (0/2, inkowalny). Przy wejściu patrzy w rękę rywala i ZMUSZA GO DO ODRZUCENIA PIEŚNI/AKCJI! "
            "W mirrorze natychmiast wyrzuca z ręki rywala jego kluczowe 'Strength of a Raging Fire'."
        )
    if "Aladdin - Doing His Part" in name:
        return (
            meta_stat
            + "2-drop 3/2 (inkowalny). Agresywne statystyki bojowe do wczesnych wymian w 2. turze."
        )
    if "You Broke My Smolder" in name:
        return (
            meta_stat
            + "Akcja za 1 ink dobierająca kartę i obniżająca siłę. Wygładza tempo na Draw i przyspiesza obieg talii."
        )
    if "Hamish, Hubert & Harris - Troublemaking Triplets" in name:
        return (
            meta_stat
            + "3-drop 3/3 za 3 ink (inkowalny). Równomierne ciało 3/3, które wymienia się 1-za-1 z większością 3-dropów."
        )
    if "Look What You've Done" in name:
        return (
            meta_stat
            + "Akcja za 2 ink (inkowalna) zadająca 2 obrażenia. Tani ping likwidujący agresywne 1-dropy."
        )
    if "Strike a Good Match" in name:
        return (
            meta_stat
            + "Pieśń za 2 ink (dobierz 2, odrzuć 1). Wymienia martwe uninkables na świeże karty i wyszukuje 'Strength'."
        )
    if "Flynn Rider - Spectral Scoundrel" in name:
        return (
            meta_stat
            + "1-drop (1/2) z Boost 2. Po dopłaceniu 2 ink staje się 3/2 z 2 Lore, wywierając natychmiastowy zegar punktowy."
        )
    if "Benja - Guardian of the Dragon Gem" in name:
        return (
            meta_stat
            + "3-drop 2/3 za 2 Lore (inkowalny). Solidny quester niszczący przedmioty, jeśli rywal tekuje The Claw/Remote."
        )
    if "Robin Hood - Champion of Sherwood" in name:
        return (
            meta_stat
            + "Shift 3 (4/4/2). Przy wygnaniu wrogiej postaci daje +2 Lore, a sam dobiera kartę przy banishmentu."
        )
    if "Tinker Bell - Giant Fairy" in name:
        return (
            meta_stat
            + "Shift 4 / 6-drop (4/5/2). 1 obrażenie AoE w stół rywala czyści wrogie 1-dropy i osłabia wrogie 'Strength'."
        )
    if "Sudden Chill" in name:
        return (
            meta_stat
            + "Pieśń za 2 ink zmuszająca rywala do odrzucenia karty. Odcina przeciwnika od kluczowych top-endów."
        )
    if "Cursed Merfolk - Ursula's Handiwork" in name:
        return (
            meta_stat
            + "1-drop 0/1 za 2 Lore. Karze rywala odrzuceniem karty, jeśli zdecyduje się go wyzwać."
        )

    return meta_stat + f"Wspiera krzywą kosztu ({card.cost} ink) w turniejowym Green Steelu."


def build_inkdecks_curated_packages(
    core_counts: dict[int, int],
    db: CardDatabase,
) -> list[tuple[str, dict[str, int], str]]:
    """
    Tworzy 12 pakietów flex (po 20 kart) opartych na autentycznych wariantach
    talii Emerald/Steel z czołówki turniejów Inkdecks.com.
    """
    return [
        (
            "Inkdecks Meta Leader (Darkwing Crime Fighter + Launchpad + Broadway + Shadowy Superhero)",
            {
                "Darkwing Duck - Crime Fighter": 4,
                "Launchpad - Trusty Sidekick": 3,
                "Darkwing Duck - Shadowy Superhero": 3,
                "Broadway - Sturdy and Strong": 2,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
            },
            "Najpopularniejsza turniejowa lista z Inkdecks (Set 14). Tani Darkwing 2/2 likwiduje wczesny stół rywala, "
            "Launchpad 2/3 dobiera karty co turę, a Broadway i Shadowy Superhero dominują tury 3-5.",
        ),
        (
            "Anti-Song & Hand Disruption (Lenny Toy Binoculars + Darkwing Duck + Strike a Good Match)",
            {
                "Lenny - Toy Binoculars": 3,
                "Darkwing Duck - Crime Fighter": 4,
                "Strike a Good Match": 3,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
                "Angel - Experiment 624": 2,
            },
            "Lenny wchodzi w 3. turze i wyrzuca rywalowi z ręki jego 'Strength of a Raging Fire'! "
            "Darkwing czyści wrogie 1-dropy, a Strike filtruje naszą rękę.",
        ),
        (
            "Heavy Midrange Bodybuilders (Broadway 4/6 + The Huntsman 4/3 + Darkwing Duck 2/2)",
            {
                "Broadway - Sturdy and Strong": 3,
                "The Huntsman - On the Queen's Orders": 3,
                "Darkwing Duck - Crime Fighter": 4,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
                "Mushu - Stealthy Dragon": 2,
            },
            "Maksymalna siła bojowa na stole. Huntsman (4/3) i Broadway (4/6) nie mają sobie równych "
            "w wymianach naziemnych, deklasując wrogie Flashe i Toddy.",
        ),
        (
            "Direct Damage Burn (The Terror That Flaps + He Hurled His Thunderbolt + Darkwing)",
            {
                "The Terror That Flaps in the Night": 3,
                "He Hurled His Thunderbolt": 3,
                "Darkwing Duck - Crime Fighter": 4,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
                "Angel - Experiment 624": 2,
            },
            "Potężny arsenał celowanych obrażeń z Inkdecks. Likwiduje postacie rywala na odległość bez ryzyka "
            "utraty własnych ciał w walkach.",
        ),
        (
            "Bounce & Tempo Denial (Rapunzel Tower Defender + Darkwing + Tod & Copper)",
            {
                "Rapunzel - Tower Defender": 3,
                "Darkwing Duck - Crime Fighter": 4,
                "Tod & Copper - Best of Friends": 3,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
                "Angel - Experiment 624": 2,
            },
            "Rapunzel cofa wysokie Shifty i drogie postacie rywala (Elinor, The Queen) niszcząc ich całą inwestycję ink, "
            "podczas gdy Tod & Copper questuje bez przeszkód w powietrzu.",
        ),
        (
            "Evasive Aerial Dominance (Tod & Copper + White Rabbit + Pegasus + Flynn Rider)",
            {
                "Tod & Copper - Best of Friends": 3,
                "Flynn Rider - Spectral Scoundrel": 4,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "Mushu - Stealthy Dragon": 2,
                "The Queen - Devious Disguise": 2,
                "Angel - Experiment 624": 2,
                "Angela - Night Warrior": 1,
            },
            "Maksymalizacja lataczy Evasive (Tod & Copper, Rabbit, Pegasus, Mushu). Uziemione postacie rywala "
            "nie mogą ich dotknąć, a gra zamienia się w jednostronny wyścig lore.",
        ),
        (
            "Card Advantage & Pure Cycling (Launchpad + Strike a Good Match + You Broke My Smolder)",
            {
                "Launchpad - Trusty Sidekick": 3,
                "Strike a Good Match": 3,
                "You Broke My Smolder": 3,
                "Darkwing Duck - Crime Fighter": 4,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 2,
                "The Queen - Devious Disguise": 2,
            },
            "Absolutna dominacja surowcowa. Launchpad i tanie akcje dobierające sprawiają, że nigdy nie brakuje "
            "odpowiedzi na stół, a rywal w turach 5-7 zostaje bez kart w ręce.",
        ),
        (
            "Heavy Ward & Untargetable (Woolter & Jesse 4x + Angela 4x + Broadway 3x)",
            {
                "Woolter & Jesse - Bellwether's Henchmen": 2,  # up to 4x total
                "Angela - Night Warrior": 1,                 # up to 4x total
                "Broadway - Sturdy and Strong": 3,
                "Darkwing Duck - Crime Fighter": 4,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
                "Mushu - Stealthy Dragon": 2,
            },
            "Kombinacja WARD i wysokiego Willpower: Woolter i Angela są całkowicie odporni na wrogie 'Strength of a Raging Fire', "
            "a Broadway jest zbyt gruby, by go dobić.",
        ),
        (
            "Shift Blowout (Robin Hood Champion + Tinker Bell Giant Fairy + Darkwing)",
            {
                "Robin Hood - Champion of Sherwood": 2,
                "Tinker Bell - Giant Fairy": 2,
                "Darkwing Duck - Crime Fighter": 4,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
                "Angel - Experiment 624": 2,
                "Launchpad - Trusty Sidekick": 2,
            },
            "Potężne Shifty: Robin Hood wygrywa grę przy banishmentu (+2 lore i karta), a Tinker Bell zamiata wrogie 1-dropy.",
        ),
        (
            "Early Suppression (Darkwing Duck + Look What You've Done + The Terror That Flaps)",
            {
                "Darkwing Duck - Crime Fighter": 4,
                "Look What You've Done": 3,
                "The Terror That Flaps in the Night": 3,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
                "Angel - Experiment 624": 2,
            },
            "Agresywna kontrola wczesnych tur: likwiduje każdego 1- i 2-dropa przeciwnika natychmiast po wejściu.",
        ),
        (
            "Hand Starvation & Disruption (Lenny Toy Binoculars + Sudden Chill + Cursed Merfolk)",
            {
                "Lenny - Toy Binoculars": 3,
                "Sudden Chill": 3,
                "Cursed Merfolk - Ursula's Handiwork": 3,
                "Darkwing Duck - Crime Fighter": 4,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 2,
                "The Queen - Devious Disguise": 2,
            },
            "Wypstrykuje przeciwnika z kart w turach 2-4. Odrzuca jego pieśni i niszczy plan na późną fazę gry.",
        ),
        (
            "Baseline Mirror Reference (Wersja Wyjściowa: 15 kart x 4 kopie)",
            {
                "Flynn Rider - Spectral Scoundrel": 4,
                "Pegasus - Gift for Hercules": 3,
                "White Rabbit - Late Again": 3,
                "The Queen - Devious Disguise": 2,
                "Mushu - Stealthy Dragon": 2,
                "Angel - Experiment 624": 2,
                "Woolter & Jesse - Bellwether's Henchmen": 2,
                "Max Goof - Rebellious Teen": 1,
                "Angela - Night Warrior": 1,
            },
            "Wersja wyjściowa (punkt odniesienia / baseline).",
        ),
    ]


def main() -> None:
    args = parse_args()
    workers_arg = args.workers if args.workers > 0 else None

    print("=" * 105)
    print("LORCANA ALPHA — TESTOWANIE REALNYCH TECHÓW Z INKDECKS.COM W MIRRORZE GREEN/STEEL")
    print("=" * 105)
    print("Pobieranie statystyk turniejowych z Inkdecks (https://inkdecks.com/lorcana-metagame/core/emerald-steel/stats)...")
    cache_path = PROJECT_ROOT / "data_cache" / "inkdecks_emerald_steel_stats.json"
    if args.refresh_inkdecks and cache_path.exists():
        cache_path.unlink()

    inkdecks_cards = scrape_inkdecks_emerald_steel(cache_path)
    print(f"[OK] Wczytano {len(inkdecks_cards)} kart z bazy Inkdecks.")

    inkdecks_lookup = {c["name"]: c for c in inkdecks_cards}

    core_counts = get_green_steel_core_counts(GLOBAL_CARD_DB)
    baseline_deck = get_green_steel_baseline_deck(GLOBAL_CARD_DB)

    print("\nKonfiguracja rozgrywki:")
    print("  Gracz 2 (Rywal):  Zablokowana lista bazowa 60 kart (15 kart po 4 sztuki)")
    print("  Gracz 1 (Test):   Zablokowany 40-kartowy rdzeń + 20 modyfikowanych slotów flex/tech z Inkdecks")
    print(f"  Gry Faza 1 (Tech): {args.games_tech} gier na kartę | Faza 2 (Pakiety): {args.games_packages} gier na pakiet")
    print(f"  Głębokość agenta:  depth={args.lookahead_depth} (Crackback) | Ziarno CRN: {args.seed}")
    print("=" * 105)

    optimizer = GreenSteelMirrorOptimizer(
        db=GLOBAL_CARD_DB,
        workers=workers_arg,
        lookahead_depth=args.lookahead_depth,
    )

    # Candidate tech names: All top competitive Inkdecks cards + under-capped core cards
    candidate_tech_names = [
        # Real Inkdecks External Techs (Presence >= 1.0% in tournaments)
        "Darkwing Duck - Crime Fighter",
        "Broadway - Sturdy and Strong",
        "Launchpad - Trusty Sidekick",
        "The Terror That Flaps in the Night",
        "Darkwing Duck - Shadowy Superhero",
        "He Hurled His Thunderbolt",
        "The Huntsman - On the Queen's Orders",
        "Rapunzel - Tower Defender",
        "Aladdin - Doing His Part",
        "Tod & Copper - Best of Friends",
        "Lenny - Toy Binoculars",
        "You Broke My Smolder",
        "Hamish, Hubert & Harris - Troublemaking Triplets",
        "Look What You've Done",
        "Strike a Good Match",
        "Benja - Guardian of the Dragon Gem",
        "Flynn Rider - Spectral Scoundrel",
        "Rapunzel & Flynn Rider - Unlikely Pair",
        "Gaetan Moliere - Clever Burrower",
        "Bobby Zimuruski - Spray Cheese Kid",
        "Three Arrows",
        "Smash",
        "Fire the Cannons!",
        "Robin Hood - Champion of Sherwood",
        "Tinker Bell - Giant Fairy",
        "Sudden Chill",
        "Cursed Merfolk - Ursula's Handiwork",
        # Under-capped core cards tested at 4x
        "Pegasus - Gift for Hercules",
        "White Rabbit - Late Again",
        "The Queen - Devious Disguise",
        "Mushu - Stealthy Dragon",
        "Angel - Experiment 624",
        "Angela - Night Warrior",
        "Max Goof - Rebellious Teen",
        "Woolter & Jesse - Bellwether's Henchmen",
    ]

    base_stats = optimizer.evaluate_baseline_mirror(baseline_deck, num_games=args.games_packages, seed=args.seed)
    print("\n" + "=" * 105)
    print("BAZOWY BENCHMARK W MIRRORZE (Wersja Wyjściowa vs Wersja Wyjściowa):")
    print(
        f"  BO1 Winrate: {base_stats.overall_winrate*100:5.1f}% | "
        f"BO3 Winrate: {base_stats.bo3_winrate*100:5.1f}% | "
        f"Play WR: {base_stats.play_winrate*100:5.1f}% | "
        f"Draw WR: {base_stats.draw_winrate*100:5.1f}%"
    )
    print("=" * 105)

    print("\n" + "=" * 105)
    print("FAZA 1: EWALUACJA REALNYCH KART TECHOWYCH Z INKDECKS W MIRRORZE GREEN/STEEL")
    print("=" * 105)

    evals = optimizer.evaluate_tech_candidates(
        core_counts=core_counts,
        baseline_deck=baseline_deck,
        candidate_names=candidate_tech_names,
        num_games=args.games_tech,
        seed=args.seed,
    )

    for idx, te in enumerate(evals, 1):
        ink_str = "Inkable" if te.inkable else "Uninkable"
        s1 = "+" if te.delta_bo1_winrate >= 0 else ""
        s3 = "+" if te.delta_bo3_winrate >= 0 else ""
        meta_info = inkdecks_lookup.get(te.card_name, {})
        pres_str = f"Inkdecks: {meta_info.get('presence', 'N/A')}"
        copies_str = f"Śr. kopii: {meta_info.get('copies', 'N/A')}"
        tag = "[Rdzeń -> 4x]" if te.is_core_tech else "[Inkdecks Tech]"

        card_tpl = GLOBAL_CARD_DB.get_by_name(te.card_name)
        pres_float = meta_info.get("presence_pct", 0.0)
        avg_float = meta_info.get("avg_copies", 0.0)
        custom_role = explain_inkdecks_mirror_role(card_tpl, pres_float, avg_float) if card_tpl else te.mirror_tactical_role

        print(
            f"{idx:2d}. {tag} {te.card_name} ({te.cost} ink, {ink_str}, {te.card_type}) — {pres_str}, {copies_str} | Testowano: {te.copies_tested}x\n"
            f"    Mirror BO1 WR: {te.bo1_winrate*100:5.1f}% ({s1}{te.delta_bo1_winrate*100:5.1f}% vs baza) | "
            f"BO3 WR: {te.bo3_winrate*100:5.1f}% ({s3}{te.delta_bo3_winrate*100:5.1f}%) | "
            f"Play: {te.play_winrate*100:5.1f}% | Draw: {te.draw_winrate*100:5.1f}%\n"
            f"    Rola taktyczna: {custom_role}"
        )

    print("\n" + "=" * 105)
    print("FAZA 2: RANKING PAKIETÓW FLEX (20 KART) ZBUDOWANYCH NA BAZIE INKDECKS")
    print("=" * 105)

    packages_def = build_inkdecks_curated_packages(core_counts, GLOBAL_CARD_DB)
    top_packages = optimizer.evaluate_packages(
        core_counts=core_counts,
        baseline_deck=baseline_deck,
        packages=packages_def,
        num_games=args.games_packages,
        seed=args.seed,
    )

    for pkg in top_packages:
        ps1 = "+" if pkg.delta_bo1_winrate >= 0 else ""
        ps3 = "+" if pkg.delta_bo3_winrate >= 0 else ""
        best_tag = " ★ [WYBÓR #1 - NAJLEPSZY PAKIET MIRROROWY]" if pkg.rank == 1 else ""
        print(
            f"\n[PAKIET #{pkg.rank:2d}]{best_tag} {pkg.package_name}\n"
            f"  Mirror BO1 WR: {pkg.bo1_winrate*100:5.1f}% ({ps1}{pkg.delta_bo1_winrate*100:5.1f}% vs baza) | "
            f"BO3 WR: {pkg.bo3_winrate*100:5.1f}% ({ps3}{pkg.delta_bo3_winrate*100:5.1f}%)\n"
            f"  Play WR: {pkg.play_winrate*100:5.1f}% | Draw WR: {pkg.draw_winrate*100:5.1f}% | "
            f"Uninkables: {pkg.uninkable_count:2d} | Średni koszt: {pkg.average_cost:.2f}\n"
            f"  Karty pakietu ({sum(pkg.flex_cards.values())} kart): "
            f"{', '.join(f'+{cnt}x {cname}' for cname, cnt in pkg.flex_cards.items())}\n"
            f"  Rola taktyczna: {pkg.tactical_summary}"
        )

    best_pkg = top_packages[0]
    tot_d1 = best_pkg.bo1_winrate - base_stats.overall_winrate
    tot_d3 = best_pkg.bo3_winrate - base_stats.bo3_winrate
    s_tot1 = "+" if tot_d1 >= 0 else ""
    s_tot3 = "+" if tot_d3 >= 0 else ""

    print("\n" + "=" * 105)
    print(
        f"SZCZEGÓŁY ZWYCIĘSKIEJ TALII (#1) W MIRRORZE GREEN/STEEL NA BAZIE INKDECKS\n"
        f"(BO1 Mirror WR: {best_pkg.bo1_winrate*100:.1f}% [{s_tot1}{tot_d1*100:.1f}% vs baza {base_stats.overall_winrate*100:.1f}%] | "
        f"BO3 Mirror WR: {best_pkg.bo3_winrate*100:.1f}% [{s_tot3}{tot_d3*100:.1f}% vs baza {base_stats.bo3_winrate*100:.1f}%])"
    )
    print("=" * 105)
    for cname, cnt in best_pkg.flex_cards.items():
        print(f"  + {cnt}x {cname}")

    print("-" * 105)
    print(
        f"GOTOWA TALIA 60 KART ({best_pkg.deck.name} | "
        f"Uninkables: {best_pkg.deck.uninkable_count} | Avg Cost: {best_pkg.deck.average_cost:.2f}):"
    )
    print(best_pkg.deck.to_dreamborn_text())
    print("=" * 105)

    # Save exports
    deck_out_path = PROJECT_ROOT / "decks" / "emerald_steel_inkdecks_mirror_optimized.txt"
    deck_out_path.parent.mkdir(parents=True, exist_ok=True)
    deck_out_path.write_text(best_pkg.deck.to_dreamborn_text() + "\n", encoding="utf-8")

    json_out_path = RESULTS_DIR / "inkdecks_green_steel_mirror_report.json"
    json_out_path.parent.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "archetype_name": "Emerald/Steel Green Steel Inkdecks Mirror",
        "source_url": "https://inkdecks.com/lorcana-metagame/core/emerald-steel/stats",
        "seed": args.seed,
        "lookahead_depth": args.lookahead_depth,
        "baseline_bo1_winrate": base_stats.overall_winrate,
        "baseline_bo3_winrate": base_stats.bo3_winrate,
        "baseline_play_winrate": base_stats.play_winrate,
        "baseline_draw_winrate": base_stats.draw_winrate,
        "optimized_bo1_winrate": best_pkg.bo1_winrate,
        "optimized_bo3_winrate": best_pkg.bo3_winrate,
        "delta_bo1_winrate": best_pkg.delta_bo1_winrate,
        "delta_bo3_winrate": best_pkg.delta_bo3_winrate,
        "selected_flex_package": best_pkg.flex_cards,
        "top_flex_packages": [
            {
                "rank": p.rank,
                "package_name": p.package_name,
                "flex_cards": p.flex_cards,
                "bo1_winrate": p.bo1_winrate,
                "bo3_winrate": p.bo3_winrate,
                "delta_bo1_winrate": p.delta_bo1_winrate,
                "delta_bo3_winrate": p.delta_bo3_winrate,
                "play_winrate": p.play_winrate,
                "draw_winrate": p.draw_winrate,
                "uninkable_count": p.uninkable_count,
                "average_cost": p.average_cost,
                "tactical_summary": p.tactical_summary,
                "dreamborn_decklist": p.deck.to_dreamborn_text(),
            }
            for p in top_packages
        ],
        "top_tech_cards": [te.__dict__ for te in evals],
        "dreamborn_decklist": best_pkg.deck.to_dreamborn_text(),
    }
    json_out_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Generate sample replays
    replays: list[str] = []
    for sp, sp_label in ((0, "P1 ON THE PLAY"), (1, "P1 ON THE DRAW")):
        replays.append(
            f"=== INKDECKS MIRROR MATCH REPLAY ({sp_label}) ===\n"
            + simulate_and_format_replay(
                deck_a=best_pkg.deck,
                deck_b=baseline_deck,
                starting_player=sp,
                seed=args.seed + sp * 777,
                lookahead_depth=max(1, args.lookahead_depth),
            )
        )
    replay_path = LOG_DIR / "inkdecks_green_steel_mirror_sample_games.log"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text("\n\n" + ("=" * 80) + "\n\n".join(replays) + "\n", encoding="utf-8")

    print(f"\n[OK] Zapisano zoptymalizowaną talię turniejową do: {deck_out_path}")
    print(f"[OK] Zapisano raport JSON do: {json_out_path}")
    print(f"[OK] Zapisano zapisy rozgrywek (Play-by-Play Replay) do: {replay_path}")


if __name__ == "__main__":
    main()
