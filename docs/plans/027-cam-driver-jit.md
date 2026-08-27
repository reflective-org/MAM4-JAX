# Plan 027 — CAM driver phase-B: `jit` + `lax.scan`

**Status:** DONE (2026-08-26; started 2026-08-26, owner: "Can you work on the
optimization and make the next PR?"). Stacked on `feat/cam-driver`
(plan 025); PR targets that branch while merges to `main` are held.

## Goal

`cam_run_step` / `cam_run_timesteps` are phase-A eager Python (ADR-004:
correctness before optimization; plan 025 G4b/G5 established correctness).
Make them compiled, mirroring what M6 PR-J1/J2 did for the E3SM driver:

1. `cam_run_step` → thin public wrapper (resolves `topology=None` OUTSIDE
   the trace — `get_topology()` raises inside a jit trace by design) over a
   jitted inner with the code-path selectors static
   (`topology`, `strat`, `n_substeps`, `do_*`, `bug_compat_stale_dumfac`,
   `reseed_dgnwet_each_step`) and the numbers traced
   (`so2_to_h2so4_rate`, everything in `state` — ADR-020's split).
   The substep loop becomes `lax.scan`; the per-substep reseed/first-step
   logic becomes a traced boolean carried as scan `xs`, so both reseed
   modes share one compiled body.
2. `cam_run_timesteps` → jitted `lax.scan` over steps (static `n_steps`),
   stacked trajectory as today. The carry pytree must be stable: the
   wrapper pre-populates `qaerwat` (zeros) the way the E3SM driver
   pre-populates calcsize's derived keys.

## Verify

1. Every existing test passes UNCHANGED — the G5 bars (per-tracer 2e-6
   print floor; totS 1e-13) have orders of headroom over XLA fusion noise.
   → `pytest tests/test_cam_driver.py` + full suite.
2. jit-cache semantics locked by tests, per the #65/#77 precedents:
   a different `so2_to_h2so4_rate` VALUE must NOT retrace (traced leaf);
   a different topology / `n_substeps` MUST hit a different cache entry
   (static); scan-trajectory ≡ repeated `cam_run_step` (consistency).
3. Measured wall-time before/after on the G5 workload (120 steps × 16
   substeps), recorded in PROGRESS.md.

## Out of scope

`vmap`/sharding (no batched host yet); making `mam_microphysics_cam` and
below individually jitted (they compile as part of the driver body);
any physics or API change beyond the wrapper/inner split.

## Results (2026-08-26)

- All existing tests pass unchanged; full G5 parity bars intact.
- `lax.scan`-over-steps trajectory is **bit-identical** to repeated
  `cam_run_step` calls (same jitted body, asserted with
  `assert_array_equal`).
- Cache semantics locked by `test_rate_is_traced_and_substeps_are_static`:
  rate sweep reuses one compilation; a new `n_substeps` retraces.
- **Measured** (G5 workload: 120 steps × 16 substeps, strat, cam_mam4):
  eager 81.6 s → jitted ~1.0 s cold (compile + run) / **0.02 s warm** —
  ~4000× warm. The trajectory scan traces the step body once regardless
  of `n_steps`.
