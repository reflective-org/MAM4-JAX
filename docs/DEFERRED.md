# Deferred

Things explicitly **not** being done now, with a brief note on **why**. Anything listed here should either (a) move to `PLANS.md` when its time comes, or (b) be deliberately retired with a one-line note.

The point of this file is to keep the *decided to skip for now* knowledge out of people's heads.

---

## License selection

- **Status:** deferred. **Tracked as issue [#47](https://github.com/reflective-org/MAM4-JAX/issues/47).**
- **Why:** No license file at the repo root yet (`README.md` notes this explicitly). The vendored Fortran reference retains its upstream license under `mam4-original-src-code/LICENSE`. Choosing a license for the JAX port (MIT? Apache-2.0? BSD-3-Clause?) is a decision that needs the owner + reflective-org alignment.
- **Resurface when:** before announcing the repo publicly or accepting external contributions.

## Stripping `__pycache__/*.pyc` from the vendored Fortran subtree

- **Status:** deferred. **Tracked as issue [#49](https://github.com/reflective-org/MAM4-JAX/issues/49).**
- **Why:** The frozen snapshot includes two `.pyc` files at `mam4-original-src-code/postprocess/postprocess_scripts/__pycache__/`. They are upstream artifacts, not part of the Fortran reference proper. They were committed verbatim to preserve snapshot fidelity (ADR-001). Removing them is a one-line change but crosses the "don't modify the vendored snapshot" boundary; better to handle it as a deliberate, single-purpose PR than a drive-by.
- **Resurface when:** the first time someone refreshes the snapshot from upstream (handle as part of the refresh), or sooner if it bothers anyone.

## JAX package scaffolding

- **Status:** deferred until owner approves Milestone 1 in `PLANS.md`.
- **Why:** Three architectural decisions still need owner input (tracer representation, process signature style, configuration mechanism — see `ARCHITECTURE.md` "Open architectural questions"). Scaffolding without those decisions risks rework.
- **Resurface when:** ADR-007 through ADR-009 are landed in `KEY_DECISIONS.md`.

## CI / continuous integration

- **Status:** deferred. **Tracked as issue [#46](https://github.com/reflective-org/MAM4-JAX/issues/46).**
- **Why:** No JAX code or tests exist yet; nothing meaningful to run in CI beyond markdown lint. Premature.
- **Resurface when:** Milestone 1 (JAX package scaffold) lands a first test.
- **Activated 2026-05-28** by owner directive — slot between M8 and M9 so CI catches regressions while cloud-chem porting lands. The resurfacing condition above was technically met since M3.6 (72-test suite exists); what changed today is the priority slotting, not the trigger.

## Differentiability through MAM4

- **Status:** **resolved 2026-05-28** by M6 PR-J5 ([#44](https://github.com/reflective-org/MAM4-JAX/pull/44)) — the diffrax-branch JAX-side is end-to-end autodiff-clean (`jax.grad` returns finite, deterministic cotangents through 60-step `scan`). Operationalizing this into a calibration demo is M9 (milestone [#4](https://github.com/reflective-org/MAM4-JAX/milestone/4)).
- **Why (original):** A motivation for porting to JAX is potential autodiff through aerosol microphysics. However, several MAM4 routines use bisection, conditional branches on physical regimes, and other constructs that don't admit clean gradients out of the box. Promising differentiability up front would be a claim we cannot yet support.
- **Resurface when:** Milestone 6 (audit + optimization), as part of the differentiability audit subtask. *(Resolved; entry retained for history.)*

## Stress validation of `calcsize_sub`'s bounds-adjust and Aitken↔accum transfer branches

- **Status:** deferred (coverage gap in the captured reference). **Tracked as issue [#48](https://github.com/reflective-org/MAM4-JAX/issues/48).**
- **Why:** Empirical inspection of a 60-step instrumented run of the box model shows that, given the namelist defaults in `mam4-original-src-code/run_test.csh` and the dgnum / sigmag in `box_model_utils/rad_constituents.F90:167-170`, two `modal_aero_calcsize_sub` branches are **never triggered**:
  1. **Number-tracer bounds adjustment** (Fortran 3-step procedure around `num_a0 → num_a1 → num_a2 → num_a3`). The per-mode `v2ncur` stays inside `voltonumblo_amode` / `voltonumbhi_amode` throughout the run, so the relaxation branch is dead code in the captured `.npz` reference.
  2. **Aitken ↔ accumulation mode transfer.** The Aitken mean diameter never grows above its `dgnumhi` and the accumulation mean diameter never drops below its `dgnumlo`, so the transfer block is also dead in the reference.
  Concretely: across 60 timesteps, all four mode-number tracers (`q[17, 22, 30, 34]`) show **exactly zero** change between `calcsize_before` and `calcsize_after`. The only thing `calcsize_sub` actually does in our reference is recompute `dgncur_a` from updated mass mixing ratios + fixed number (max rel-change ~5.5e-2 in `dgncur_a`).
  The M3.5 port still implements both branches faithfully (rule #6), but the .npz-based regression test cannot **directly** catch a bug in those branches — they will appear identical to a no-op port for these inputs. We accept the coverage gap because (a) the JAX port is a line-by-line transcription and (b) any future workflow that does perturb the state into the adjust/transfer regime will surface bugs as soon as it's exercised.
- **Resurface when:** any of these triggers:
  - A multi-day box-model run (or a synthetic spin-up) drives the mode sizes out of `[dgnumlo, dgnumhi]` and exercises the transfer / adjust branches in a captured reference.
  - We add a synthetic test fixture in `tests/test_calcsize.py` with `q` and `num` values that intentionally violate bounds (e.g., manually set `num` such that `v2ncur > voltonumbhi`).
  - The first downstream code path that actually depends on the adjust/transfer outputs lands (e.g., a multi-step amicphys + calcsize loop).

## Porting the standalone `modal_aero_{rename,gasaerexch,newnuc,coag}.F90` modules

- **Status:** explicitly **not** planned for the box-model port.
- **Why:** Discovered during M3.5 PR-B planning that these standalone modules are **not invoked** by the box-model driver. `driver.F90:1283` calls `modal_aero_amicphys_intr`, which contains its own self-contained copies of all four sub-processes (`mam_gasaerexch_1subarea`, `mam_rename_1subarea`, `mam_newnuc_1subarea`, `mam_coag_1subarea`) plus orchestration. The standalone modules are reachable only via a different call graph (`modal_aero_rename_sub` is called solely from `modal_aero_gasaerexch.F90:685`, which itself is unreachable from this driver). See `docs/ARCHITECTURE.md` "amicphys is self-contained" for the full module map.
  - The M3 plan therefore targets the `mam_*_1subarea` versions inside `modal_aero_amicphys.F90`.
  - Numerical results from a port of the standalones could differ from the active code path even if the algorithms are conceptually the same (different code, different rounding paths, potentially different defaults).
- **Resurface when:** any of:
  - The active call graph changes (e.g., we adopt a different MAM driver where `modal_aero_gasaerexch.F90`'s entry point becomes live).
  - An academic/research interest justifies porting the standalone implementation alongside (e.g., to compare it against the active path or to support a different host model in the future).

## Multi-column / multi-level support

- **Status:** deferred. **Now promoted to a forward milestone — M12** (milestone [#7](https://github.com/reflective-org/MAM4-JAX/milestone/7)). M5 has been green-partial on `main` since 2026-05-22 and M6 PR-J3 verified `vmap`-cleanness on `diffrax`, so the structural prerequisites are met. Status here remains "deferred" until M12 moves from proposed → in progress.
- **Why:** The Fortran reference is configured with `PCOLS=1`, `PVER=1` (single-column, single-level). The JAX port targets that configuration first. Generalizing to multiple columns/levels via `vmap` is straightforward but not the validation target.
- **Resurface when:** after Milestone 5 (convergence test reproduction) is green. *(Condition met; see M12.)*

## GPU / TPU sharding

- **Status:** deferred. **Now promoted to a forward milestone — M13** (milestone [#8](https://github.com/reflective-org/MAM4-JAX/milestone/8)). Originally PR-J6 in M6's plan; broken out into its own milestone (2026-05-28) because the work is substantially distinct from M6's JIT/scan/vmap/grad cleanup.
- **Why:** Validation is the priority; CPU `float64` is the easiest path to a clean Fortran diff. Sharding decisions belong in Milestone 6 once correctness is established.
- **Resurface when:** Milestone 6. *(Condition met; see M13.)*

## Adaptive sub-stepping in `_mam_soaexch_1subarea` (permanently deferred on `main`; resolved on `diffrax` branch per ADR-013)

- **Status:** **permanently deferred on `main`** (decision 2026-05-22, ADR-013). Resolved on the long-lived `diffrax` branch where diffrax's standard adaptive controller provides substepping for free.
- **Why:** M5 (12-point convergence sweep, 2026-05-22) confirmed Fortran's adaptive substepping fires at `dt ≥ 60s`, causing 6 of 12 sweep cases to diverge from `main`'s JAX port at up to 1.3e-1 rel-err. The original plan was a handwritten PR-E2 to close the gap. Owner reframing (2026-05-22): the migration to diffrax (Milestone 7) provides adaptive substepping natively, so handwritten substepping in `main` would duplicate work that diffrax replaces. Instead, M7 lives on a parallel `diffrax` branch and resolves these cases there.
- **What this means on `main`:**
  - `tests/test_sweep.py::test_sweep_xfail_without_adaptive_soa_substep[1..30]` stays `xfail` indefinitely on `main`. Docstring points at the diffrax branch.
  - No PR-E2 is planned for `main`.
- **Resurface when:** the project reverses course on ADR-013, OR a downstream need forces small-`dt` accuracy in `main` specifically (e.g., a multi-column fixture where Fortran's substepping fires at a `dt` where it doesn't on the box-model). Until then, `diffrax` is the resolution path.

## Diffrax migration for the handwritten solvers (Milestone 7)

- **Status:** core PRs done (2026-05-26 through 2026-05-28) on the long-lived `diffrax` branch per ADR-013. PR-I1 / PR-D1 / PR-D2 landed; PR-D3 permanently deferred (entry below). **Merge-back to `main` per ADR-016 is in progress** via the merge-back PR (opened 2026-05-28, reverses the 2026-05-28 morning's brief deferral). See `docs/PLANS.md` Milestone 7 + GitHub milestone [#2](https://github.com/reflective-org/MAM4-JAX/milestone/2) for the sub-PR catalog.
- **Why:** the handwritten H₂SO₄ analytical solver (PR-D), soaexch step-1/step-2 semi-implicit (PR-E), and coag's analytical solvers (PR-G) are good candidates for replacement by [`diffrax`](https://github.com/patrick-kidger/diffrax) (JIT/grad/vmap-clean, better stiff-system numerics, adaptive stepping for free). Migrating on `main` would change Fortran-vs-JAX output by ~1 ULP (different solver tolerances), which complicates the 1e-6 bit-validation baseline; ADR-013 resolves this by keeping diffrax on its own branch where the baseline is "match `main` within ADR-003" instead of "match Fortran".
- **Resurface in `main` only if:** ADR-013 is revisited (the merge-back convention is currently deferred, not cancelled — could resume once the owner directs).

## Coag analytical → diffrax (PR-D3)

- **Status:** **permanently deferred** on the `diffrax` branch (2026-05-26). PR-I1, PR-D1, PR-D2 of M7 are complete; PR-D3 was conditional on a motivating issue surfacing during PR-D1/PR-D2, and none did.
- **Why coagulation isn't worth porting to diffrax:**
  - **Coagulation has no differential equation in time on the box-model substep.** `_mam_coag_1subarea` applies *closed-form algebraic formulas* derived from analytically integrating the coagulation kinetics over one substep: a two-branch number-loss guard at `tmpa < 1e-5` and `(1 − exp(−tmpb))` mass-transfer formulas per active pair. There is no per-substep ODE state to integrate, no internal solver decisions for diffrax to make. Replacing the closed form with `solve_ivp` reformulates an algebraic step as a numerical-ODE step — strictly more expensive per call (Kvaerno5 takes 5–7 RHS stages × adaptive internal substeps × implicit-solver iterations vs ~140 LOC of jnp ops in the handwritten code), with no accuracy benefit.
  - **No validation gap.** PR-D1's mass-conservation trace + PR-D2's ablation experiment confirmed that total SO4 mass and aerosol number conserve to ~ε between JAX and Fortran. Coag is bit-clean against the Fortran reference; there is no rel-err to close.
  - **No autodiff motivation on this fixture.** The existing handwritten coag is fully JIT-clean and `vmap`-clean. The two-branch number-loss has a non-smooth gradient at the `tmpa = 1e-5` boundary, but no downstream consumer in scope cares about that smoothness. If a future autodiff use case (e.g., gradient-based calibration of a coag-sensitive parameter) materialises and the gradient discontinuity matters, this entry resurfaces.
  - **No stiffness motivation.** Coag's closed-form is mathematically exact for the underlying kinetics over one substep; no stiff regime has surfaced where the closed form fails. ADR-015's deferral condition ("a motivating stiffness or autodiff issue") has not been met.
- **Resurface if:** (a) a downstream autodiff use case requires smoother gradients across the `tmpa = 1e-5` boundary, or (b) the box-model configuration changes to a fixture where the closed-form formulas lose accuracy (e.g., very stiff coagulation regimes the current closed-form doesn't capture).

---

*When adding a new deferred item: state what, why, and the condition that would bring it back.*
