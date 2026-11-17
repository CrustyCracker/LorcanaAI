# Core / flex scan smoke

Core N=44 (`decks/amber_emerald_princess_aggro_core44.txt`), flex=16  
Baseline: `decks/amber_emerald_princess_aggro_optimized.txt`  
Engine: Python DSL (`use_full_dsl_sim=True`), lookahead=0, top-meta=3, games=4, packages=2, tech-candidates=8, seed=42

> Smoke tylko — wyniki głośne. Po smoku przywrócono oryginalny `amber_emerald_princess_aggro_optimized.txt` (optimize-core nadpisuje ten plik).

## Top techs (Phase 1 external)

| # | Tech | Rec. | ΔBO1 | ΔBO3 | Best MU signal |
|--:|------|------|-----:|-----:|----------------|
| 1 | Nani - Stage Manager | 3x | +22.9% | +26.7% | vs Emerald/Steel Mid |
| 2 | David - Protective Snowboarder | 3x | +15.2% | +15.2% | vs Emerald/Steel Mid |
| 3 | Rapunzel - Tower Defender | 2x | +7.6% | +3.8% | vs Emerald/Steel Mid |
| 4 | Lionheart - Incumbent Mayor | 2x | +7.6% | +3.8% | vs Emerald/Steel Mid |
| 5 | Percy - Pupsicle | 3x | +2.5% | +2.5% | vs Circle of Life |

## Top flex package (smoke #1)

+3 Percy - Pupsicle, +2 Pocahontas - Finding the Way, +4 Finnick - Tiny Terror, +4 Mickey Mouse - Expedition Leader, +2 Pluto - Friendly Pooch, +1 Lionheart - Incumbent Mayor  
(Smoke BO3 ≈ +26.7% vs baza — **nie** traktować jako finalnego rankingu.)

## Targeted scan-tech (budget 12)

vs Emerald/Steel Midrange: Top line `+2 Ohana Means Family` (out Finnick) — przy budget 12 Δ≈0 (szum).

## Re-run (produkcja)

```bash
python lorcana_cli.py sync-meta --period week --top-n 15

python lorcana_cli.py optimize-core \
  --archetype "Amber/Emerald Princess Aggro" \
  --core-file decks/amber_emerald_princess_aggro_core44.txt \
  --baseline-file decks/amber_emerald_princess_aggro_optimized.txt \
  --top-meta 8 --games 48 --packages 16 --tech-candidates 36 --lookahead-depth 1

# Athena (duży batch)
sbatch slurm/optimize_princess_aggro_athena.sbatch
```
