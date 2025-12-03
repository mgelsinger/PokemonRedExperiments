# PokemonRedExperiments – Project Status (Dec 2025)

## What works

- End-to-end PPO training pipeline runs without crashing.
- Environment wrapper around Pokémon Red emulator is functional.
- TensorBoard logging is wired up for:
  - Episode statistics (`rollout/ep_len_mean`, `rollout/ep_rew_mean`, etc.).
  - Reward components (`reward_components/*`).
  - Env stats and battle stats (`env_stats/*`, `battle_stats/*`).
- The agent reliably:
  - Survives full-length episodes (hitting the 2047-step cap).
  - Explores the overworld and triggers some scripted events.

## What we observed

- Episodes almost always reach max length:
  - `episode/length_mean ≈ 2047`, `length_min == length_max`.
- Shaped reward increases over time:
  - `reward_components/total_shaped` and `rollout/ep_rew_mean` trend upward.
  - Reward is dominated by `exploration` and `milestone`, not battles.
- Battle and progression statistics remain effectively zero:
  - `battle_stats/wins_mean`, `battle_stats/losses_mean`, `battle_stats/total_mean` all ~0.
  - `env_stats/badge = 0`, `success_rate = 0`, `max_map_progress = 1`.
  - `levels_sum_mean` remains low and unstable.

In short: the agent learns to **walk around and farm shaping**, but not to **play Pokémon** (no consistent battling, leveling, or badges).

## Why this is hard

- Pokémon Red is a huge, long-horizon RL problem:
  - Very large state space (JRPG with menus, overworld, battles).
  - Sparse, delayed rewards (badges, late-game progress).
  - Many steps of “junk” behavior (walking, bumping, menu navigation) before meaningful events.
- With the current setup:
  - Environment is too large / open-ended.
  - Reward shaping strongly favors exploration over battles / badges.
  - Battle and progression events are too rare for PPO to learn from effectively.

## Future work (if resumed)

If this project is revived in the future, useful next steps would be:

1. **Small-map or battle-heavy environment**
   - Restrict early gameplay to a tiny region (e.g., Pallet + Route 1) where battles are frequent.
   - Or build a battle-only sub-environment with no overworld navigation.

2. **Reward shaping redesign**
   - Strongly reward:
     - Starting battles, winning battles, gaining levels, earning badges, hitting key flags.
   - Down-weight:
     - Raw exploration and “just walking around.”
   - Add configuration for reward weights to allow experiments without code changes.

3. **Curriculum**
   - Train separate:
     - Battle-only agent (learn combat).
     - Nav-only agent (learn to reach goals).
   - Then integrate into a small-map environment and expand gradually.

4. **Better metrics and evaluation**
   - Log and track:
     - Battles per episode, wins per episode.
     - Badges, map progress, deaths.
   - Add regular evaluation runs and checkpointing based on these metrics.

## Current decision

As of now, this repository is considered **“paused / experimental”**.

The groundwork (emulator integration, environment, training loop, logging) is valuable, but the remaining work to get a truly capable Pokémon-playing agent is research-level and beyond the intended scope for now.

A new project will focus on a more tractable RL domain (e.g., a Mario-style platformer), where:
- Reward is denser and more directly aligned with progress.
- Episodes are shorter and easier to evaluate.
- Users can more easily train and compare agents.
