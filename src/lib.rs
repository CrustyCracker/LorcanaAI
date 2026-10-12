use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_pcg::Pcg64Mcg;
use rayon::prelude::*;

/// Compact representation of a card template for high-speed Rust rollouts on PLGrid Athena (128-core EPYC)
#[derive(Clone, Copy, Debug)]
pub struct FastCardSpec {
    pub card_id: u16,
    pub cost: u8,
    pub inkable: bool,
    pub is_character: bool,
    pub is_song: bool,
    pub strength: i16,
    pub willpower: i16,
    pub lore: u8,
    pub flags: u32, // Bitflags: 1=Rush, 2=Ward, 4=Evasive, 8=Bodyguard, 16=Singer, 32=Removal, 64=Draw
    pub effect_value: i16,
}

#[derive(Clone, Copy, Debug)]
pub struct FastBoardChar {
    pub card_id: u16,
    pub strength: i16,
    pub willpower: i16,
    pub damage: i16,
    pub lore: u8,
    pub exerted: bool,
    pub drying: bool,
    pub flags: u32,
}

/// Fast rollout simulation for a matchup between two 60-card decks (represented as arrays of compact card specs)
fn simulate_single_fast_game(
    deck1: &[FastCardSpec],
    deck2: &[FastCardSpec],
    starting_player: usize,
    seed: u64,
) -> (usize, u32, i32, i32) {
    let mut rng = Pcg64Mcg::new(seed as u128 | 1);
    let mut d1 = deck1.to_vec();
    let mut d2 = deck2.to_vec();
    d1.shuffle(&mut rng);
    d2.shuffle(&mut rng);

    let mut hands: [Vec<FastCardSpec>; 2] = [d1.split_off(d1.len().saturating_sub(7)), d2.split_off(d2.len().saturating_sub(7))];
    let mut decks: [Vec<FastCardSpec>; 2] = [d1, d2];

    // Tactical Mulligan: keep low-cost inkables (cost <= 3), mulligan expensive or excess uninkables
    for p in 0..2 {
        let mut kept = Vec::with_capacity(7);
        let mut put_back = Vec::new();
        let mut uninkable_kept = 0;
        for c in hands[p].drain(..) {
            if c.cost <= 3 && (c.inkable || uninkable_kept < 2) {
                if !c.inkable {
                    uninkable_kept += 1;
                }
                kept.push(c);
            } else {
                put_back.push(c);
            }
        }
        let draw_cnt = put_back.len();
        for _ in 0..draw_cnt {
            if let Some(drawn) = decks[p].pop() {
                kept.push(drawn);
            }
        }
        decks[p].extend(put_back);
        decks[p].shuffle(&mut rng);
        hands[p] = kept;
    }

    let mut lore: [i32; 2] = [0, 0];
    let mut ink_total: [u8; 2] = [0, 0];
    let mut board: [Vec<FastBoardChar>; 2] = [Vec::new(), Vec::new()];
    let mut turn_count: u32 = 0;
    let mut active = starting_player % 2;

    while turn_count < 80 {
        turn_count += 1;
        let opp = 1 - active;

        // Ready step + passive location/character lore
        for ch in board[active].iter_mut() {
            ch.exerted = false;
            ch.drying = false;
        }

        // Draw step (skip on very first turn of starting player)
        if turn_count > 1 {
            if let Some(card) = decks[active].pop() {
                hands[active].push(card);
            } else {
                // Decked out loss
                return (opp, turn_count, lore[0], lore[1]);
            }
        }

        // 1. Ink step: choose highest-cost inkable card not playable this turn
        if let Some((idx, _)) = hands[active]
            .iter()
            .enumerate()
            .filter(|(_, c)| c.inkable)
            .max_by_key(|(_, c)| c.cost)
        {
            if ink_total[active] < 9 || hands[active].len() > 2 {
                hands[active].swap_remove(idx);
                ink_total[active] += 1;
            }
        }

        let mut avail_ink = ink_total[active];

        // 2. Play cards greedily by tactical impact
        hands[active].sort_by(|a, b| b.cost.cmp(&a.cost));
        let mut i = 0;
        while i < hands[active].len() {
            let c = hands[active][i];
            if c.cost <= avail_ink {
                avail_ink -= c.cost;
                hands[active].remove(i);
                // Check draw flag (64)
                if (c.flags & 64) != 0 {
                    let draw_n = (c.effect_value.max(1) as usize).min(2);
                    for _ in 0..draw_n {
                        if let Some(dc) = decks[active].pop() {
                            hands[active].push(dc);
                        }
                    }
                }
                // Check removal/damage flag (32)
                if (c.flags & 32) != 0 && !board[opp].is_empty() {
                    if let Some((t_idx, _)) = board[opp]
                        .iter()
                        .enumerate()
                        .filter(|(_, oc)| (oc.flags & 2) == 0) // Not Ward
                        .max_by_key(|(_, oc)| oc.lore as i16 * 3 + oc.strength)
                    {
                        board[opp][t_idx].damage += c.effect_value.max(2);
                        if board[opp][t_idx].damage >= board[opp][t_idx].willpower {
                            board[opp].swap_remove(t_idx);
                        }
                    }
                }
                if c.is_character {
                    board[active].push(FastBoardChar {
                        card_id: c.card_id,
                        strength: c.strength,
                        willpower: c.willpower.max(1),
                        damage: 0,
                        lore: c.lore,
                        exerted: false,
                        drying: (c.flags & 1) == 0, // Rush (1) skips drying for challenge
                        flags: c.flags,
                    });
                }
            } else {
                i += 1;
            }
        }

        // 3. Quest vs Challenge decisions
        let opp_lore_threat: i32 = board[opp].iter().map(|c| c.lore as i32).sum();
        let my_potential_lore: i32 = board[active]
            .iter()
            .filter(|c| !c.exerted && !c.drying && (c.flags & 1) == 0)
            .map(|c| c.lore as i32)
            .sum();

        for idx in 0..board[active].len() {
            let ch = board[active][idx];
            if ch.exerted {
                continue;
            }
            let can_quest = !ch.drying;
            let can_challenge = !ch.drying || (ch.flags & 1) != 0;

            // If we have lethal lore on board, always quest!
            if can_quest && (lore[active] + my_potential_lore >= 20) {
                lore[active] += ch.lore as i32;
                board[active][idx].exerted = true;
                if lore[active] >= 20 {
                    return (active, turn_count, lore[0], lore[1]);
                }
                continue;
            }

            // Look for favorable challenge against exerted opponent character
            let mut challenged = false;
            if can_challenge && ch.strength > 0 {
                if let Some((t_idx, _)) = board[opp]
                    .iter()
                    .enumerate()
                    .filter(|(_, oc)| {
                        oc.exerted
                            && ((oc.flags & 4) == 0 || (ch.flags & 4) != 0) // Evasive check
                    })
                    .filter(|(_, oc)| {
                        let kills_opp = (oc.damage + ch.strength) >= oc.willpower;
                        let survives = (ch.damage + oc.strength) < ch.willpower;
                        kills_opp && (survives || oc.lore >= 2 || opp_lore_threat + lore[opp] >= 16)
                    })
                    .max_by_key(|(_, oc)| oc.lore as i16 * 4 + oc.strength)
                {
                    board[active][idx].exerted = true;
                    let opp_str = board[opp][t_idx].strength;
                    board[opp][t_idx].damage += ch.strength;
                    board[active][idx].damage += opp_str;
                    if board[opp][t_idx].damage >= board[opp][t_idx].willpower {
                        board[opp].swap_remove(t_idx);
                    }
                    challenged = true;
                }
            }

            if !challenged && can_quest && ch.lore > 0 {
                lore[active] += ch.lore as i32;
                board[active][idx].exerted = true;
                if lore[active] >= 20 {
                    return (active, turn_count, lore[0], lore[1]);
                }
            }
        }

        // Remove banished characters on active side
        board[active].retain(|c| c.damage < c.willpower);
        active = opp;
    }

    let winner = if lore[0] >= lore[1] { 0 } else { 1 };
    (winner, turn_count, lore[0], lore[1])
}

/// Parallel batch simulation across all available CPU threads (Rayon) on PLGrid Athena
#[pyfunction]
fn simulate_batch_matchup_rs(
    deck1_tuples: Vec<(u16, u8, bool, bool, bool, i16, i16, u8, u32, i16)>,
    deck2_tuples: Vec<(u16, u8, bool, bool, bool, i16, i16, u8, u32, i16)>,
    num_games: usize,
    base_seed: u64,
) -> PyResult<(usize, usize, usize, usize, f64)> {
    let d1: Vec<FastCardSpec> = deck1_tuples
        .into_iter()
        .map(|t| FastCardSpec {
            card_id: t.0,
            cost: t.1,
            inkable: t.2,
            is_character: t.3,
            is_song: t.4,
            strength: t.5,
            willpower: t.6,
            lore: t.7,
            flags: t.8,
            effect_value: t.9,
        })
        .collect();

    let d2: Vec<FastCardSpec> = deck2_tuples
        .into_iter()
        .map(|t| FastCardSpec {
            card_id: t.0,
            cost: t.1,
            inkable: t.2,
            is_character: t.3,
            is_song: t.4,
            strength: t.5,
            willpower: t.6,
            lore: t.7,
            flags: t.8,
            effect_value: t.9,
        })
        .collect();

    let results: Vec<(usize, usize, u32)> = (0..num_games)
        .into_par_iter()
        .map(|i| {
            let start_p = i % 2;
            let (winner, turns, _, _) =
                simulate_single_fast_game(&d1, &d2, start_p, base_seed.wrapping_add(i as u64 * 9973));
            (start_p, winner, turns)
        })
        .collect();

    let mut d1_wins_play = 0usize;
    let mut d1_games_play = 0usize;
    let mut d1_wins_draw = 0usize;
    let mut d1_games_draw = 0usize;
    let mut total_turns = 0u64;

    for (start_p, winner, turns) in results {
        total_turns += turns as u64;
        if start_p == 0 {
            d1_games_play += 1;
            if winner == 0 {
                d1_wins_play += 1;
            }
        } else {
            d1_games_draw += 1;
            if winner == 0 {
                d1_wins_draw += 1;
            }
        }
    }

    let avg_turns = if num_games > 0 {
        total_turns as f64 / num_games as f64
    } else {
        0.0
    };

    Ok((d1_wins_play, d1_games_play, d1_wins_draw, d1_games_draw, avg_turns))
}

#[pymodule]
fn lorcana_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_batch_matchup_rs, m)?)?;
    Ok(())
}
