# Part 2: analyses and numbers removed from the docs

Removed because no tracked analysis produces them (the lead's rule, 2026-10-26: replace with the computed
value, remove analyses that were never computed, and say so). Replacements of typed numbers by computed ones
are in `changes.md`.

| Doc | Claim / section | Removed | Why |
|---|---|---|---|
| `paper/claims_table.md` | 15 | "duration-controlled, within-animal (GEE ×4.91, 5/5)" | No tracked analysis fits the duration-controlled GEE; the 2026-10-12 refresh review lists it as not re-derived. Plain medians and the animals in direction stand in its place |
| `paper/claims_table.md` | 16b | "(all 5/5, duration-controlled, GEE p<1e-100)" | Same: no tracked duration-controlled GEE |
| `paper/claims_table.md` | 16, 16b, 17 (Comparison) | "duration-controlled" | Same |
| `paper/claims_table.md` | 2c framing row | "speed … ×4.9"; "Duration-controlled and 5/5 within animals where tested" | Same |
| `paper/claims_table.md` | 18 | "duration is also the key confound controlled in #15–#17" | Nothing controls it; now says #15–#17 are not duration-controlled |
| `paper/claims_table.md` | 7b | the random-position reference ("~71 cm random"), the verdict "not GPI/baseline-driven", and "validated on one session/animal (5/5)" | The 242-epoch spike-evidence run computes no random-position reference (`GPI_REMOTE_CM` is defined in `analyze_event_spike_evidence.py` and never used); without it the distance alone does not show the decode is not a ground-process default |
| `paper/claims_table.md` | 23 | per-animal ranges of the median non-local share (Kay 86–91 %, Karlsson 73–82 %, Yu 74–85 %, MUA 78–84 %, Carey 80–94 %) | Superseded by the pooled medians and the Local / No-Spike split; not removed for lack of a producer |

Wording changed to fit the computed numbers (not removals): claim 14 no longer says non-SWR content
"represents reward-well locations (predominantly remote wells)" or is "well-dominated" (45 % remote well
against 41 % remote path); claim 17's multimodality is no longer "weak, not robust" (4/5 animals).
