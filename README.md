# LorcanaAlpha: System AI do Gry, Analizy Matchupów i Budowania Talii (PLGrid Athena Edition)

Kompletny system sztucznej inteligencji dla gry **Disney Lorcana**, łączący:
1. **Ultraszybki Silnik Gry i Kompilator Efektów DSL** ([`LorcanaSimulator`](lorcana_ai/engine/simulator.py), [`EffectDSLCompiler`](lorcana_ai/data/effect_dsl.py) + natywny akcelerator Rust `PyO3 + Rayon` w [`src/lib.rs`](src/lib.rs)).
2. **Dwupoziomowego Agenta Grającego** ([`ISMCTSAgent`](lorcana_ai/agents/is_mcts.py), [`LorcanaTransformerNet`](lorcana_ai/agents/networks.py) oraz [`OpponentBeliefNetwork`](lorcana_ai/agents/networks.py) przewidujący ukrytą rękę i archetyp rywala).
3. **Trójmodułowy Meta-Optymalizator Talii**:
   - **Moduł 1: Tech-Card Scanner ([`TechCardScanner`](lorcana_ai/deckbuilder/tech_scanner.py))** – algorytm *Multi-Armed Bandit (UCB1)* szukający najlepszych kart (1–4 kopii) na konkretny matchup (z podziałem na *Play* / *Draw* i *Bo3*).
   - **Moduł 2: MAP-Elites Deckbuilder ([`MAPElitesDeckbuilder`](lorcana_ai/deckbuilder/evolutionary_map.py))** – koewolucja genetyczna odkrywająca nowe talie we wszystkich 15 parach kolorów Ink i 3 profilach (*Aggro*, *Midrange*, *Control*).
   - **Moduł 3: Nash Meta-Solver ([`NashMetaSolver`](lorcana_ai/deckbuilder/nash_solver.py))** – wyliczanie równowagi Nasha metagame turniejowego w formacie *Best-of-3* oraz generator poradników Mulliganu.

---

## Szybki Start (Lokalnie na Windows / Linux)

```powershell
# 1. Instalacja zależności Python
pip install -r requirements.txt

# 2. (Opcjonalnie) Pobranie najnowszej pełnej bazy LorcanaJSON do pamięci podręcznej
python lorcana_cli.py sync-cards

# 3. Uruchomienie testów weryfikacyjnych całego systemu
python tests/test_all.py

# 4. Skanowanie najlepszych Tech-Kart i strategii Mulliganu na konkretny matchup
python lorcana_cli.py scan-tech --my-deck "Amethyst/Ruby Bounce Control" --opp-deck "Ruby/Sapphire Ramp Control" --copies 3 --budget 96

# 5. Odkrywanie nowych talii za pomocą MAP-Elites
python lorcana_cli.py evolve --generations 4 --offspring 6 --games 8

# 6. Wyliczenie równowagi Nasha dla metagame turniejowego Bo3
python lorcana_cli.py nash-meta --games 12

# 7. Uruchomienie interaktywnego Dashboardu Webowego (Tech Scanner, MAP-Elites, Gra vs AI)
streamlit run dashboard/app.py
```

---

## Uruchamianie na klastrze PLGrid Athena (ACK Cyfronet)

1. Skopiuj katalog `AGENTS` na swój obszar `$SCRATCH` lub `$HOME` na klastrze **Athena**.
2. Na węźle dostępowym (*login node*, gdzie działa internet) wykonaj jednorazową inicjalizację (instalacja środowiska, kompilacja natywnego silnika Rust `lorcana_rs` z instrukcjami `AVX2/EPYC` oraz pobranie bazy kart offline):
   ```bash
   bash slurm/setup_athena_env.sh
   ```
3. Zleć trening sieci neuronowej (Transformer Policy-Value + Opponent Belief Network) w trybie Self-Play na kartach **NVIDIA A100** (z automatycznym zapisywaniem i wznawianiem checkpointów):
   ```bash
   sbatch slurm/train_agent_a100.sbatch
   ```
4. Zleć masową ewolucję talii MAP-Elites i skanowanie Tech-Kart na 64–128 rdzeniach **AMD EPYC**:
   ```bash
   sbatch slurm/evolve_decks_athena.sbatch
   ```
