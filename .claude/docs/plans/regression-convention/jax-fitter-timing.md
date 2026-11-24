# NumPy fitter against a JAX prototype: how long whole workloads take (2026-11-23)

A copy of `.claude/research/2026-11-20-regression-prototypes/jax_fitter/bench/REPORT.md` (main checkout,
not tracked), copied into the plan on 2026-11-23. The lead read this table and decided on it (design,
section 9, twenty-third round). Written by the main session from the benchmark worker's hand-back.
Scripts, in that research folder: `jaxfit.py` (the prototype), `bench.py`, `summarize.py`, `run_*.sh`. Raw
results and every per-case table: `results/` and `results/tables.md` there. No repository file was edited.
All counts are simulated.

**Every timing was taken under load** (load average 37 to 80 on 96 logical CPUs, mostly other users' jobs).
The same JAX-only work run 5 minutes apart differed by 2.7 times. Read the absolute times as rough. The ratios
of JAX to NumPy are more reliable, because both ran in turn in the same process on the same cores.

## Whole workloads

| Workload | Fits | NumPy, CPU time on 1 core | NumPy, wall on 32 processes | JAX CPU, CPU time on 1 core | JAX CPU, wall on 32 processes | JAX GPU, wall on 1 card (float64) |
|---|---|---|---|---|---|---|
| Real rate fit: 242 x 2 kinds x 2 models x 3 penalties | 2,904 | 16.9 min | 32 s | 35.4 min | 1.4 min | 5.2 min |
| One penalty calibration: 18 x 242 x 2 x 4 terms | 34,848 | 3.4 h | 6.3 min | 7.0 h | 13.5 min | 60 min |
| One simulation sweep: 6 x 40 x 242 x 2 x 2 | 232,320 | 22.5 h | 42 min | 46.9 h | 1.5 h | 6.6 h |
| Content fit: 242 x 2 x 3 | 1,452 | 23 s | 1 s | 36 s | 4 s | 46 s |
| mPFC, per 1,000 units (40,000 rows x 60 columns) | 1,000 | 311 s | 10 s | 198 s (batched, 1 thread) | 6 s | 0.8 s (one batch) |
| mPFC at repository size (extrapolated) | 166,968 | 99 h | 3.1 h | 63 h | 2.0 h | 15 min |

Per-fit times behind the table (30 real sessions fitted in one process, NumPy and JAX in turn, scaled to the
cohort's mean number of rows): 0.349 s (NumPy), 0.727 s (JAX on CPU), 0.102 s (one GPU), for a whole fit at
fixed penalties (checks, rotation, Newton, covariance, effective parameters). JAX compile cost: 15.5 s per
process (13 array shapes). Wall time on 32 processes assumes perfect scaling; it was not measured. The mPFC
size comes from `paper/reported/mpfc.json` (`model_fits`, `bins_fit`), used only as a size.

## What the numbers say

1. On the CPU, JAX is about 2 times slower than NumPy for a rate fit (1.6 to 2.1 on one thread, about 1.5 on
   8 threads). The cost is XLA's float64 `X'WX`, which is slower than OpenBLAS.
2. On one GPU, a rate fit is about 3 times faster than NumPy on one core. The Newton part takes 11 to 13 ms on
   the GPU against 210 to 300 ms in NumPy; the rest of the GPU's 85 to 134 ms is NumPy bookkeeping before the
   fit (column cut, rotation, checks), which depends only on the design and could be done once per session
   (not measured).
3. For many fits that share one design (the mPFC units), batching wins on every device: per unit 311 ms in a
   NumPy loop, 198 ms (1 thread) or 41 ms (8 threads) in a JAX batch on CPU, 2.2 ms (100 units) or 0.8 ms
   (1,000 units) on the GPU. Peak GPU memory 4.7 GB by `nvidia-smi`.
4. Compile time is small against any whole workload: 0.6 to 3.8 s per new shape, 15.5 s per process; a filled
   persistent cache removes it.
5. The content model is small with every option (under a minute).

## Correctness of the prototype against the NumPy fitter

- Statuses and Newton step counts equal in every case, CPU and GPU.
- Rate fits: coefficients differ by at most 2.9e-12 (normwise relative); covariance 3.6e-12; effective
  parameters 5.5e-12. Elementwise relative differences are below 1e-9 except one coefficient near zero
  (about 4e-4 in size) at 5.4e-9; it passes a tolerance with an absolute floor (1e-11 + 1e-9 |b|).
- Content and mPFC cases: about 1e-16 normwise.
- GPU against CPU: 5.5e-13 to 8.8e-13 for rate.

## Notes for the port

- JAX 0.8.0 has no flag for the number of CPU threads: its thread pool follows the process's CPU set, so a
  pool of fitting processes must pin each process (`taskset` or CPU affinity). This corrects the port design,
  section 8.
- GPU run: card 9 on zephyr, about 7 minutes, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, float64 by the scoped
  `enable_x64`; no process left on the card. Another user's jobs took that card shortly after.

## Not measured

Refit rounds after "no finite maximum"; penalty selection by REML; the bookkeeping done once per session;
real scaling over 32 processes; several processes per GPU or several cards; bit-for-bit GPU repeats; mPFC at
its real size; any timing on an idle machine; rate-model simulation sweeps batched over data sets (an
inference from the mPFC case, see the main session's note to the lead).
