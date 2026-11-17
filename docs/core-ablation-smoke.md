# CoreValue ablation smoke

Deck: `decks/amber_emerald_princess_aggro_optimized.txt`  
Core: `decks/amber_emerald_princess_aggro_core44.txt` (N=44, flex=16)  
Engine: `MatchupEvaluator(use_full_dsl_sim=True)`, lookahead=0, games/MU=4, top-meta=4, seed=42  
Mode: 59-card (−1 copy, no `--replace`)

Baseline meta: BO1 **63.4%** | BO3 **69.1%**

## Weakest → strongest core cards (by ΔBO3 after −1 copy)

| # | Card | x | ΔBO1 | ΔBO3 | Abl BO3 |
|---|------|--:|-----:|-----:|--------:|
| 1 | Bobby Zimuruski - Spray Cheese Kid | 4 | +4.3% | +7.4% | 76.5% |
| 2 | Elinor - Renowned Diplomat | 4 | +2.3% | +0.1% | 69.2% |
| 3 | Ursula - Deceiver | 4 | −1.8% | −1.6% | 67.4% |
| 4 | Mickey Mouse - Best in Town | 4 | −2.0% | −2.0% | 67.0% |
| 5 | Pocahontas - Peacekeeper | 4 | −6.1% | −8.1% | 61.0% |
| 6 | Dale - Ready for His Shot | 4 | −9.1% | −12.6% | 56.5% |
| 7 | Thomas - Wide-Eyed Recruit | 4 | −16.0% | −18.6% | 50.5% |
| 8 | Grandmother Willow - Ancient Advisor | 4 | −10.9% | −22.4% | 46.7% |
| 9 | Mike Wazowski - Heroic Climber | 4 | −21.4% | −24.8% | 44.2% |
| 10 | Aurora - Holding Court | 4 | −20.4% | −29.7% | 39.4% |
| 11 | Pocahontas - Guiding the Tribe | 4 | −31.8% | −44.7% | 24.4% |

**Najsłabsze (smoke):** Bobby Zimuruski, Elinor.  
**Najmocniejsze (smoke):** Pocahontas - Guiding the Tribe, Aurora, Mike Wazowski.

Szum oczekiwany przy 4 grach/MU — do decyzji deckbuildingowych: `--games 24+ --lookahead-depth 1 --top-meta 8`.

```bash
python lorcana_cli.py core-ablate \
  --deck decks/amber_emerald_princess_aggro_optimized.txt \
  --core-deck decks/amber_emerald_princess_aggro_core44.txt \
  --vs-meta week --top-meta 8 --games 24 --lookahead-depth 1
```

JSON: `results/core_ablation_report.json`
