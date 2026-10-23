# Spatial Bandit Task: Experimental Description

Detailed description of the behavioral task, training, and data collection
for the "Spatial Bandit" experiment analyzed in this project.

Source: Comrie, Monroe, Kahn, Denovellis, Joshi, Guidera, Krausz, Berke,
Daw, & Frank (2024). *Hippocampal representations of alternative
possibilities are flexibly generated to meet cognitive demands.* bioRxiv
preprint,
[doi:10.1101/2024.09.23.613567](https://doi.org/10.1101/2024.09.23.613567)
(Version 1, posted 2024-09-23).

## Task Overview

The Spatial Bandit task is a dynamic foraging task designed so that
performance benefits from representing alternative possibilities — both
for deliberating among alternatives and for updating information about
them. It combines spatial memory with probabilistic decision-making.

### Maze geometry

- Three radially arranged Y-shaped **foraging patches** emerging from a
  center, forming a "three-legged spider" track.
- Each patch has a central hallway that bifurcates into two hallways,
  each terminating in a photogated **reward port**.
- 6 reward ports total (2 per patch), 3 patches.
- Hallways are 6.5 cm wide, linear segments 53 cm long, 120° between
  arms (three-fold symmetry).
- Black acrylic track with 3 cm walls; elevated, in a dimly lit room
  with distal spatial cues.
- See [CLAUDE.md](../CLAUDE.md#spatial-bandit-task-structure) for the
  track graph node/edge numbering used in analysis code.

### Reward structure

- Each port delivers 100 μL of liquid food reward (evaporated milk with
  5% sucrose) via syringe pump, gated by an infrared beam break at the
  port.
- Reward is **probabilistic**. In the **canonical main protocol** each port's
  nominal p(R) ∈ {0.2, 0.5, 0.8}, one patch has a higher *average* p(R) than the
  other two (the "best patch"), and the best patch's two ports are (0.5, 0.8).
- **The one hard, always-enforced constraint is that no single patch is (0.8, 0.8).**
  It is *not* true that only the best patch contains an 0.8 port: **non-best patches
  frequently hold an 0.8** (e.g. a (0.2, 0.8) patch, average 0.5) — "best patch =
  (0.5, 0.8)" means the *highest-average* patch, not the only 0.8-bearing one.
  (Verified from the rig-side per-session Python control files that set `leafProbs`,
  `/stelmo/alison/<animal>/<animal>_py_done/*.py`: across 352 contingencies, `(0.2,0.8)`
  patches occur 188× and `(0.8,0.8)` occurs 0×. The shipped `qvals_table` agrees:
  no `(0.8,0.8)` patch on any of its 105 359 stops.)
- **Dataset caveat (matters for value/contingency covariates).** The recorded and
  ingested epochs are **not** all canonical-protocol sessions. Two different
  questions have two different answers, and they are often confused:

  - *Do the contingencies follow the canonical rule?* **No, for 35.6 % of stops**
    (37.4 % of the 473 distinct epoch-label × contingency combinations in that
    table; the table carries no animal/day key, so that is the coarsest honest
    block unit). Every one of those failures is
    the same failure: no patch is (0.5, 0.8), so the "best patch = (0.5, 0.8)"
    clause does not hold. By stop share the top patch is (0.2, 0.8) for 19.4 %,
    (0.5, 0.5) for 8.4 %, (0.2, 0.5) for 7.1 %, and the all-p=1.0 shaping
    configuration for 0.8 %; 2.5 % of stops have two patches tied for the highest
    mean. The always-enforced (0.8, 0.8) prohibition holds on **100 %** of stops.
  - *Do the contingencies use the canonical values?* **Yes, for 99.2 % of stops** —
    all six port probabilities drawn from {0.2, 0.5, 0.8}. The residual 0.8 %
    (838 stops) is the single all-p=1.0 shaping configuration.

  So the non-canonical mass is overwhelmingly **structural** — early-training
  configurations built from canonical values but without the best-patch structure —
  not exotic probabilities. Both figures were measured on the shipped
  `Processed-Data/replay_content/qvals_table.parquet`
  (105 359 stops; `task_reward_probability_port_{0..5}`).

  The rig-side control files (`/stelmo/alison/<animal>/<animal>_py_done/*.py`) gave the
  original ~37 % figure (chimi + peanut ground truth) and agree, but they are a
  **broader parse**: they cover every session the rig ran, including ones never
  ingested. Special sessions using
  probabilities outside {0.2, 0.5, 0.8} (e.g. {0.1, 0.3, 0.4, 0.6, 0.9}) appear
  there and **not** in the ingested cohort — which is why the raw-file view and the
  table view disagree about value diversity while agreeing about the rule.

  Either way, candidate-arm reward-belief mean, task reward probability, and
  `task_reward_probability_patch_mean_gap_best_vs_second` are computed over a mix of
  canonical and non-canonical contingencies unless sessions are explicitly filtered
  (they are **not** currently — only technical exclusions apply; see
  `data_loaders/constants.py`).
- **Consecutive pokes at the same port are never rewarded** — animals
  must alternate between ports to receive reward.
- p(R) assignments are uncued; animals can only infer them through
  experience across multiple visits.

### Trial and block structure

- A **trial** is defined as the period from the **final photogate exit** at one
  port to the final photogate exit at another (different) port. Consecutive
  re-pokes at the same port remain one task port experience; the destination
  clock is its first poke-in and the departure clock is its final poke-out.
  This matches the StateScript parsing in Comrie's analysis and encompasses
  both the run between ports and the reward outcome at the destination.
- **Contingency blocks**: reward probabilities change covertly every 60
  trials (n=4 animals) or 80 trials (n=1 animal), or at a 20-minute
  time cap (rarely hit).
- Block transitions change which patch is best, and are
  pseudorandomized to counterbalance:
  - which patch is best,
  - which port within the best patch is higher value,
  - clockwise vs. counterclockwise ordering of best/medium/worst
    patches around the track.
- **Run sessions**: 180 trials (n=4 animals) with 3 contingencies × 60
  trials, or 160 trials (n=1 animal) with 2 contingencies × 80 trials.
- **Daily structure**: 7 or 8 run sessions (~20 min each) interleaved
  with ≥30-minute rest sessions in a rest box. Each day starts and ends
  with a rest session.
- Data collected over 8–17 days per animal.

### First-session exposure

The first run session for each animal is atypical: p(R) = 1 at all
ports for the first 100 trials, then p(R) = 0.5 at all ports for the
next 100 trials. This familiarized animals with the environment and
probabilistic rewards before the main protocol began.

## Trial classifications

- **Stay trial**: animal visits a different port *within the same
  patch*.
- **Switch trial**: animal visits a port *in a different patch*.
- Animals tend to alternate within a patch (Stay) and occasionally move
  to an alternative patch (Switch), consistent with patch foraging.

## Behavioral characterization

Animals learn to concentrate visits in the highest-p(R) patch and adapt
to uncued block changes; see Comrie et al. 2024 for the behavioral
learning model and Stay/Switch value analyses on this dataset.

## Subjects

- 5 adult male Long-Evans rats (Charles River), 450–650 g.
- Pair housed pre-training, single housed post-training with food
  restriction to 85% baseline weight.
- 12-hr light/dark cycle (lights 6 AM – 6 PM).

## Pre-training

1. Linear track with walls, alternating between two ports for liquid
   reward: two 40-min sessions + one 25-min session.
2. Elevated linear track (1.1 m long, 84 cm high): to criterion of ≥30
   rewards in a 15-minute session.
3. Pre-training environment is fully separate from the Spatial Bandit
   room.
4. Return to ad libitum food before surgery; refamiliarized with the
   elevated track post-surgery recovery. Two rats had additional
   experience on a fork maze.

## Neural implant

Custom hybrid microdrive:

- **24 independently movable tetrodes** (12.5 μm nichrome, gold-plated
  to ~250 kΩ), 12 per hemisphere, targeting dorsal hippocampus CA1
  stratum pyramidale.
- **Polymer probes** (Lawrence Livermore, 128-channel, up to 4 stacking
  sets per animal) targeting **mPFC** and **OFC**.
- Stainless-steel ground screw over cerebellum (global reference); one
  tetrode per hemisphere advanced to corpus callosum as local reference.

### Implantation coordinates

- Dorsal HC: ±2.6–2.8 ML, −3.7 to −3.8 AP (relative to Bregma).
- Polymer probes: mPFC and OFC.

### Post-surgery

- Tetrodes advanced daily over ~3–4 weeks to CA1 pyramidal layer,
  guided by spiking and LFP signatures.
- Animals recovered, then food-restricted to 85% body weight and
  retrained on the elevated linear track before starting the Spatial
  Bandit task.

## Recording

- Continuous neural data + digital I/O (beam breaks, pumps) recorded
  with **Trodes 1.8.2 (SpikeGadgets)**.
- Sampling rate: **30 kHz** (n=4 animals) or **20 kHz** (n=1 animal).
- Behavior room: 2.4 × 2.9 m, dim, black distal cues on white walls,
  plastic black curtain separating experimenter.
- Ceiling-mounted Allied Vision camera at 30 fps, PTP-synchronized.
- Head position / direction tracked online via red+green LED ring on
  implant using SpikeGadgets Trodes.

## Task control software

- **Statescript** (SpikeGadgets) + custom Python scripts control
  behavior.
- **Trodes** (SpikeGadgets) handles neural acquisition and automated
  session control.

## Relationship to this codebase

- Terminology bridge (paper ↔ code):
  - "port" (paper) ↔ **well** (code; track-graph nodes 0–5).
  - "hallway" / "linear track segment" (paper) ↔ **arm** /
    **track segment** (code; track-graph edges 0–8).
- Each **trial** in `data["trials"]` corresponds to one port-to-port
  traversal as defined above.
- Each **well visit** corresponds to the beam-break interval at a port
  (reward delivery gated by the beam break).
- **Patch change** trials = Switch trials; within-patch trials = Stay
  trials.
- **Contingency block** boundaries are not currently exposed as a
  first-class variable in the data loader — they can be reconstructed
  from the trial sequence and port p(R) metadata when needed.
- See [CLAUDE.md](../CLAUDE.md) for how the task structure maps onto
  track-graph nodes, edges, and patch IDs used throughout the analysis
  code.
