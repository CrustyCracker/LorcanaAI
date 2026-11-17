# Play & Tech Models (LorcanaAI)

Zmienna architektura talii: **core N + flex (60−N)**. Gry i rankingi techów zawsze na **Python DSL** (`MatchupEvaluator(use_full_dsl_sim=True)`).

## Core / Flex

| Element | Opis |
|--------|------|
| Core N | Zablokowany rdzeń (np. 44 Princess → `decks/amber_emerald_princess_aggro_core44.txt`) |
| Flex 60−N | Sloty techowe skanowane UCB1 / `optimize-core` |
| CoreValue | Leave-one-out: −1 kopia karty rdzenia → ΔWR vs meta |

## Play models

| Warstwa | Rola |
|--------|------|
| **Heuristic L1 (primary)** | `TacticalHeuristicAgent` w `MatchupEvaluator` — wszystkie skany, ablacje, optimize-core |
| **IS-MCTS (validation)** | Spotcheck Top-K / watch-game; **nie** do rankingu meta |
| **LLM** | Brak — zero LLM do wyboru ruchów |

```bash
# Heuristic playtest (człowiek vs AI)
python lorcana_cli.py playtest --deck-a decks/amber_emerald_princess_aggro_optimized.txt --deck-b "Amber/Sapphire Detectives"

# Spectator heuristic / IS-MCTS
python lorcana_cli.py watch-game --deck-a ... --deck-b ... --agent heuristic
python lorcana_cli.py watch-game --deck-a ... --deck-b ... --agent is-mcts --simulations 16 --determinizations 4
```

## Tech model

- **TechCardScanner** — UCB1 bandit (`scan-tech`, `optimize-core`)
- Zawsze `use_full_dsl_sim=True` (ścieżka Rust bitflag wyłączona dla oficjalnych rankingów)
- Flex packages: kombinacje 60−N slotów vs gauntlet Duels.ink

## CoreValue ablation

```bash
python lorcana_cli.py core-ablate \
  --deck decks/amber_emerald_princess_aggro_optimized.txt \
  --core-deck decks/amber_emerald_princess_aggro_core44.txt \
  --vs-meta week --top-meta 8 --games 24 --lookahead-depth 1
```

- Domyślnie talia ma 59 kart (−1 kopia); opcjonalnie `--replace "Card Name"` wypełnia do 60.
- Najwyższy ΔBO3 = najsłabsza karta rdzenia; najbardziej ujemny = najmocniejsza.
- Smoke: `docs/core-ablation-smoke.md`

## Flex scan

```bash
python lorcana_cli.py optimize-core \
  --archetype "Amber/Emerald Princess Aggro" \
  --core-file decks/amber_emerald_princess_aggro_core44.txt \
  --baseline-file decks/amber_emerald_princess_aggro_optimized.txt \
  --top-meta 8 --games 48 --packages 16 --lookahead-depth 1
```

Smoke (szum przy małym budget): `docs/core-flex-scan-smoke.md`

## IS-MCTS spotcheck

Pełny gauntlet IS-MCTS jest za wolny do rankingów. Wrapper:

```bash
python lorcana_cli.py ismcts-spotcheck \
  --deck-a decks/amber_emerald_princess_aggro_optimized.txt \
  --deck-b "Emerald/Steel Midrange" \
  --games 2 --simulations 8 --determinizations 2

# Po scan-tech — re-score Top-K z JSON
python lorcana_cli.py ismcts-spotcheck \
  --tech-json results/latest_tech_scan.json \
  --deck-a decks/amber_emerald_princess_aggro_optimized.txt \
  --deck-b "Emerald/Steel Midrange" --top-k 3 --games 2
```

Moduł: `lorcana_ai/deckbuilder/ismcts_spotcheck.py`. Lokalnie może pojawić się ostrzeżenie NumPy/torch — spotcheck i tak kończy się na CPU.

---

## Jak testować aplikację (krok po kroku)

Skrót użytkownika: `docs/how-to-test-lorcanaai.md`.

1. **Sync meta** — `python lorcana_cli.py sync-meta --period week --top-n 15`
2. **Playtest** — `python lorcana_cli.py playtest --deck-a <twoja>.txt --deck-b "<archetyp>"`
3. **CoreValue** — `python lorcana_cli.py core-ablate --deck <60>.txt --core-deck <N>.txt --vs-meta week --games 24`
4. **Flex / tech** — `optimize-core` (cały flex) lub `scan-tech` (jeden matchup)
5. **Interpretacja** — ΔBO3 / rankingi; smoke ≠ ranking produkcyjny (podnieś `--games` / `--budget`)
6. **IS-MCTS** — tylko spotcheck Top-K lub `watch-game --agent is-mcts`
7. **Athena** — duże batche (poniżej); laptop — playtest, smoke, małe skany

---

## Athena vs local

### Co Athena realnie robi w tym repo

Athena **nie jest tylko** scaffoldem NN. Skrypty SLURM wołają ten sam `lorcana_cli.py` co lokalnie:

| Job | Plik | Co robi |
|-----|------|---------|
| Train NN (GPU A100) | `slurm/train_agent_a100.sbatch` | `lorcana_cli.py train --use-mcts` — Policy/Value + Belief |
| Evolve / Nash / scan | `slurm/evolve_decks_athena.sbatch` | `evolve`, `nash-meta`, `scan-tech` |
| Princess flex opt | `slurm/optimize_princess_aggro_athena.sbatch` | `optimize-core` + `watch-game` + `scan-tech` |
| Circle / Green Steel | `slurm/optimize_*_athena.sbatch` | to samo dla innych archetypów |
| Setup | `slurm/setup_athena_env.sh` | venv + (opcjonalnie) Rust `lorcana_rs` |

`MatchupEvaluator` / `TechCardScanner` skalują się po `SLURM_CPUS_PER_TASK` (CPU). **GPU jest wymagane tylko do `train`.** Sbatch’e optimize/evolve siedzą na partycji `plgrid-gpu-a100` głównie przez grant/account — ciężar to CPU, nie wagi sieci.

### Kiedy Athena, kiedy laptop

| Zadanie | Gdzie |
|--------|--------|
| Playtest, sync-meta, smoke (`--games 4`, `--budget 12`) | **Laptop** |
| Core-ablate / optimize-core produkcyjnie (`--games 48–1200`, wiele pakietów) | **Athena** (32–128 CPU) |
| `scan-tech` z dużym `--budget` (500+) | **Athena** |
| `train` Self-Play IS-MCTS + Transformer | **Athena A100** |
| Dashboard Streamlit | **Lokalnie** (`streamlit run dashboard/app.py`) — nie ma sbatch UI |

### Exact Athena (bez „pushowania” lokalnych modeli)

Nie musisz kopiować checkpointów, żeby robić skany/symulacje — Heuristic L1 + DSL wystarczy. Na login node:

```bash
# jednorazowo
bash slurm/setup_athena_env.sh

# ciężki flex + scan + watch-game (Princess)
sbatch slurm/optimize_princess_aggro_athena.sbatch

# trening NN (tu GPU + checkpointy w checkpoints/)
sbatch slurm/train_agent_a100.sbatch

# evolve + nash + scan
sbatch slurm/evolve_decks_athena.sbatch
```

Ręcznie na alokacji CPU (ten sam CLI co lokalnie):

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python3 lorcana_cli.py core-ablate \
  --deck decks/amber_emerald_princess_aggro_optimized.txt \
  --core-deck decks/amber_emerald_princess_aggro_core44.txt \
  --vs-meta week --top-meta 10 --games 48 --lookahead-depth 1 \
  --workers "${SLURM_CPUS_PER_TASK:-32}"

python3 lorcana_cli.py scan-tech \
  --my-deck "Amber/Emerald Princess Aggro" \
  --opp-deck "Emerald/Steel Midrange" \
  --budget 1500 --copies 3 --lookahead-depth 1
```

Env typowe: `SLURM_CPUS_PER_TASK`, `LD_LIBRARY_PATH` (libpython z modułu), `OMP_NUM_THREADS=1`, venv z `setup_athena_env.sh`. Account w sbatch: `plgvideocnn0-gpu-a100` (dostosuj do swojego grantu).

**Wniosek:** Athena hostuje train **oraz** batch `optimize-core` / `scan-tech` / `watch-game` / (nowe) `core-ablate` bez lokalnych wag. Lokalnie trzymaj playtest + małe smoke; duże rankingi odpalaj remote workerem / sbatch.
