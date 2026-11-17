# How to test LorcanaAI (guided path)

Krótka ścieżka: meta → playtest → core-ablate → scan-tech → interpretacja → Athena.

## 1. Sync meta

```bash
python lorcana_cli.py sync-meta --period week --top-n 15
```

Talie lądują w `decks/duels_ink_core_bo1/`, cache w `data_cache/duels_ink_core_bo1.json`.

## 2. Playtest

```bash
python lorcana_cli.py playtest \
  --deck-a decks/amber_emerald_princess_aggro_optimized.txt \
  --deck-b "Amber/Sapphire Detectives" --seed 2137
```

Ty = P0, heurystyka = P1. Na CHOICE wybierz numer celu. `q` = wyjście.  
Spectator: `watch-game` (opcjonalnie `--agent is-mcts`).

## 3. Core-ablate (CoreValue −1)

```bash
python lorcana_cli.py core-ablate \
  --deck decks/amber_emerald_princess_aggro_optimized.txt \
  --core-deck decks/amber_emerald_princess_aggro_core44.txt \
  --vs-meta week --top-meta 8 --games 24 --lookahead-depth 1
```

Tabela: najwyższy ΔBO3 = kandydat do cięcia z rdzenia. JSON: `results/core_ablation_report.json`.

## 4. Scan-tech / flex

```bash
# Jeden matchup (UCB1)
python lorcana_cli.py scan-tech \
  --my-deck "Amber/Emerald Princess Aggro" \
  --opp-deck "Emerald/Steel Midrange" \
  --budget 96 --copies 2 --lookahead-depth 1

# Cały flex wokół core N
python lorcana_cli.py optimize-core \
  --archetype "Amber/Emerald Princess Aggro" \
  --core-file decks/amber_emerald_princess_aggro_core44.txt \
  --baseline-file decks/amber_emerald_princess_aggro_optimized.txt \
  --top-meta 8 --games 48 --packages 16 --lookahead-depth 1
```

Zawsze Python DSL (`use_full_dsl_sim=True`).

## 5. Interpret results

- Smoke (`--games 4`, `--budget 12`) → tylko weryfikacja pipeline; rankingi głośne.
- Produkcja: `--games 24+`, `--budget 96+`, `--lookahead-depth 1`.
- CoreValue: słabe karty na górze tabeli ablacjii; silne na dole (duży ujemny Δ).
- Tech: dodatni ΔBO3 vs baza; patrz też breakdown matchupów.
- IS-MCTS spotcheck: jakość gry Top-K, **nie** ranking meta.

## 6. Kiedy Athena

- **Laptop:** sync-meta, playtest, smoke, mały scan-tech.
- **Athena:** duże `optimize-core` / `core-ablate` / `scan-tech`, `train` na A100.
- Start: `bash slurm/setup_athena_env.sh`, potem `sbatch slurm/optimize_princess_aggro_athena.sbatch` (lub własny job z tym samym CLI).
- Modele NN nie są potrzebne do skanów — wystarczy Heuristic L1.

Pełny model + Athena: `docs/play-and-tech-models.md`.
