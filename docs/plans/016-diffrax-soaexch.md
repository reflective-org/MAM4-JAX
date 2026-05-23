# Plan 016 — M7 PR-D1: port `_mam_soaexch_1subarea` to diffrax

> **Status:** proposed 2026-05-22. Awaiting owner approval before
> implementation.

---

## Context

PR-I1 (PR #31, merged 2026-05-22) put the diffrax infrastructure
in place on the `diffrax` branch:

- `diffrax>=0.7` is a dependency.
- `mam4_jax/solvers.py` exports `SolverConfig`, `SolverResult`,
  and `solve_ivp(rhs, y0, t0, t1, args, saveat, config)` — the
  latter currently raises `NotImplementedError`.
- ADR-014 sets the merge-back intent and the
  main → diffrax sync convention.

PR-D1 is the first **solver swap**. It does two coupled things:

1. **Implements `solve_ivp`** by wiring it to
   `diffrax.diffeqsolve` with `Kvaerno5` (the PR-I1 default) and a
   PI step-size controller.
2. **Replaces the handwritten `_mam_soaexch_1subarea` body** in
   `mam4_jax/processes/amicphys.py` with a call to `solve_ivp`.

This PR is the scientifically load-bearing one for M7: it is
expected to flip the 6 currently-`xfail`ed M5 sweep cases
(`nstep ∈ {1, 2, 4, 9, 18, 30}`, i.e. `dt ≥ 60 s`) to passing.
Per ADR-013's branch invariants, all already-passing tests on
`diffrax` must stay green (with ~1 ULP slack tolerated).

## Scope (this PR, `diffrax` branch only)

### `mam4_jax/solvers.py` — wire the body

Replace the `NotImplementedError` body with the diffrax call.
Proposed implementation (sketch — full version in PR):

```python
def solve_ivp(rhs, y0, t0, t1, args=None, saveat=None, config=SolverConfig()):
    solver_cls = getattr(diffrax, config.solver)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(rhs),
        solver_cls(),
        t0=t0,
        t1=t1,
        dt0=config.dt0,
        y0=y0,
        args=args,
        saveat=saveat or diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIController(
            rtol=config.rtol, atol=config.atol,
        ),
        max_steps=config.max_steps,
    )
    return SolverResult(ts=sol.ts, ys=sol.ys, stats=sol.stats)
```

Map every `SolverConfig` field to its diffrax counterpart. No
fallback paths, no try/except around `getattr` — invalid solver
names should fail loudly at construction.

Two implementation notes the soaexch port relies on:

- **`getattr(diffrax, config.solver)` only resolves top-level
  diffrax exports** (`Kvaerno5`, `Tsit5`, `Dopri5`, etc.). If a
  future call site needs `diffrax.implicit.<...>` or a similarly
  nested solver, the lookup grows a small `getattr` chain. Out
  of scope for PR-D1; flagging the limit so it doesn't surprise.
- **The default `saveat=None` collapses to `SaveAt(t1=True)` —
  endpoint only.** That is the fast path for callers that only
  need `result.ys[-1]`. The soaexch port needs both endpoints to
  form the trapezoidal `qgas_avg`, so it must pass
  `saveat=diffrax.SaveAt(t0=True, t1=True)` explicitly;
  otherwise `result.ys[0]` is not recorded.

### `mam4_jax/processes/amicphys.py` — port `_mam_soaexch_1subarea`

The current handwritten body solves a semi-implicit Step-1 / Step-2
scheme for one substep. The diffrax port integrates the same ODE
adaptively.

**ODE state vector.** A flat array of length `NTOT_AMODE + 1`
per (col, level):
- `y[0]`     = `g_soa` (gas-phase SOA, scalar)
- `y[1:5]`   = `a_soa[mode]` (per-mode aerosol-phase SOA)

Batch dimensions (col, level) pass through diffrax via `vmap` or
by leaving the leading axes intact (diffrax handles array-valued
states natively).

**RHS function.** For each mode `i` where uptake is active
(`uptkaer_soag[i] > 0`):

```
g_star[i] = (g0_soa / max(a_opoa[i] + y[1+i], A_MIN1)) * y[1+i]
flux[i]   = uptkaer_soag[i] * (y[0] - g_star[i])
dy_aero[i] = flux[i]                    # condensation onto mode i
dy_gas    -= flux[i]                    # mass conservation
```

`a_opoa`, `uptkaer_soag`, `g0_soa`, and the `skip_mode` mask are
passed via `args` (closed-over per call, constant for the
integration interval). When `skip_mode[i]` is true,
`uptkaer_soag[i] == 0` → that mode's flux is identically zero,
which is the natural way to "disable" a mode within an ODE
integrator (no `where` chain in the RHS).

**Boundary conditions / clamps.** The current handwritten port
applies `max(0, ·)` to the new gas and aerosol concentrations.
The ODE structure makes the depleted-aerosol direction
well-behaved — as `y[1+i] → 0`, `g_star[i] → 0`, so the flux
becomes `uptkaer * y[0]`, pure condensation, bounded by the
available gas. But the depleted-gas direction is **not**
math-guaranteed non-negative: when `y[0] → 0`, the flux flips
sign to `-uptkaer * g_star[i]`, driving mass out of the
aerosol, and the bound `|flux| ≤ uptkaer * g0_soa` does not by
itself prevent `a_soa[i]` from dipping below zero between
integrator steps. The handwritten port's `max(0, ·)`
post-integration clamp is therefore retained as a numerical
safety net (not a math-derived invariant); diffrax's adaptive
controller should keep drift small, but the clamp closes the
loop on rare integrator overshoots.

**`qgas_avg` (time-averaged gas).** The current port computes
this as `(qgas_prv + g_soa_new) / 2` — a trapezoidal estimate over
a single substep. With diffrax we get the real trajectory: use
`SaveAt(t0=True, t1=True)` (endpoint + initial point) and take
the trapezoidal average. If the validation residual on
`qgas_avg`-consuming downstream code (newnuc) needs more
precision, switch to a denser `SaveAt` grid and a trapezoidal
rule over the recorded points. Decision to be made empirically
in implementation.

**Solver configuration.** Per PR-I1's defaults: `Kvaerno5`,
`rtol=1e-9`, `atol=1e-12`, `dt0=None` (let diffrax pick),
`max_steps=4096`. PI controller (diffrax default). All
overridable via `SolverConfig` argument to the wrapper if a
specific call site needs different settings.

**Caller-side cleanup.** The current handwritten port has a
docstring comment ("A runtime assertion in the calling test
trips loudly if this ever fails") referring to the single-substep
assumption. That assertion (and the comment) should be removed
since adaptive substepping is what diffrax does for us.

## Validation

Two-stage acceptance:

### Stage 1: SOA-only single-toggle fixture (already exists)

`tests/reference/per_process_gasaerexch/` — captured with
`mdo_gasaerexch=1`, others off, `skip_pcarbon_aging.patch`
applied. 60 timesteps of Fortran reference at `dt=30s`.

The existing test `tests/test_amicphys.py` already exercises
this path. The acceptance bar:

- All currently-passing assertions stay passing at `rtol=1e-6`,
  with up to 1 ULP slack permitted per ADR-013.
- The max rel-err on the 4 SOA tracers
  (`h2so4_gas`-companion `soag_gas`, plus `soa_aer` in each of
  the 3 SOA-eligible modes) should be reported in the residual
  plot for visibility, even if it's well under the bar.

### Stage 2: Convergence sweep (the load-bearing one)

`tests/test_sweep.py` parametrizes 12 step counts. The 6
currently `xfail`ed cases (`nstep ∈ {1, 2, 4, 9, 18, 30}`) should
flip to expected-pass at `rtol=1e-6`. Concretely:

- Remove the `@pytest.mark.xfail` markers for those cases (or the
  parametrization that triggers them).
- Re-run the sweep. Expected outcome: 12/12 passing, worst rel-err
  similar to the `nstep ≥ 60` half (currently 1.98e-8) or
  modestly larger due to diffrax's choice of internal substeps
  at large `dt`.
- If any case stays above `rtol=1e-6` despite tightening the
  internal solver tolerances (`rtol`/`atol` in `SolverConfig`),
  treat it as a finding worth investigating before declaring
  PR-D1 done — don't relax the validation bar.

### Stage 3: Driver trajectory (existing M4 fixture)

`tests/reference/per_process_full_minus_pcarbon_aging/` — the
end-to-end 60-step trajectory used in M4 PR-B (PR #26). The
existing `tests/test_driver.py` assertions should stay green at
the same `rtol=1e-6` bar. Worst rel-err there was 1.97e-8 on
`main`; diffrax may shift this — quote both before/after numbers
in the PR description.

## Plots

- **`docs/figures/soaexch_diffrax_residuals.png`** — per-step
  rel-err on the 4 SOA tracers from the single-toggle fixture
  (Stage 1). Replaces (or sits alongside) `soaexch_residuals.png`
  from the M3.6 PR-E port.
- **`docs/figures/sweep_convergence_diffrax.png`** — refresh of
  `sweep_convergence.png` (M5 plot) with the 6 previously-`xfail`
  cases now showing real points instead of the shaded
  "PR-E2 deferred" region.

## Verification

- `python -c "from mam4_jax import solvers; r = solvers.solve_ivp(rhs=lambda t,y,args: -y, y0=1.0, t0=0.0, t1=1.0); print(r.ys[-1], r.stats)"` runs end-to-end on a trivial decay problem; result close to `exp(-1)`.
- `python -m pytest tests/test_amicphys.py` — green at `rtol=1e-6`.
- `python -m pytest tests/test_driver.py` — green at `rtol=1e-6`.
- `python -m pytest tests/test_sweep.py` — **12 passed, 0 xfailed**
  (was 6 passed / 6 xfailed). Worst rel-err for each of the 12
  step counts logged in the PR description.
- `python -m pytest tests/` (full suite on `diffrax`):
  **74 passed, 0 xfailed** (= 68 on `diffrax` post-PR-I1 + 6
  flipped xfails). If the xfail-bearing
  `test_sweep_xfail_without_adaptive_soa_substep` parametrization
  is deleted in favor of extending the main
  `test_sweep_matches_fortran` parametrization to all 12 step
  counts, the slot count is preserved (6 deleted + 6 added) so
  the total stays at 74. Any deviation from 74 means a test was
  added or removed for another reason and should be called out
  in the PR description.
- New / updated residual figures regenerable from a script under
  `scripts/`.

## What this PR does NOT do

- **No H₂SO₄ analytical-solver port** — that's PR-D2.
- **No coag analytical-solver port** — that's PR-D3 (deferred
  unless motivated).
- **No `jit` boundary changes** — Phase A. PR-D1 should be
  JIT-compatible (diffrax is JIT-clean), but the explicit `jit`
  wrapping is M6.
- **No tolerance relaxation.** The `rtol=1e-6` ADR-003 bar holds
  on every existing test. If a new fixture is needed because the
  current ones don't exercise some path, add it; don't relax the
  threshold.
- **No driver-level changes.** The state-dict contract and the
  operator-splitting loop in `mam4_jax/driver.py` are untouched.

## Open questions

- **`qgas_avg` integration strategy.** Trapezoidal over
  `[t0, t1]` (endpoint average) vs trapezoidal over a denser
  `SaveAt` grid. Cheapest correct option: start with the
  endpoint average (matches the handwritten port's `(qgas_prv +
  g_soa_new) / 2`). **Important interaction:** Fortran's
  `qgas_avg` is trapezoidal over its *adaptive* substeps, which
  at large `dt` (the very `nstep ≤ 30` cases PR-D1 is designed
  to fix) is strictly more accurate than the endpoint-only
  trapezoid. So if any of those cases stay above `rtol=1e-6`
  after this port, the cause may not be H₂SO₄ uptake alone but
  also `qgas_avg` diverging from Fortran's substep-aware
  integral. Diagnose by switching to a denser `SaveAt` grid
  before tightening the solver further.
- **Batched integration.** Diffrax accepts array-valued `y0` and
  applies the solver component-wise. Confirm during
  implementation that this works for the per-(col, level)
  batched state. If not, fall back to `vmap` over the lead axes.
- **Stiff-solver compile time and Tsit5 dry-run.** `Kvaerno5`
  is implicit; the JIT compile may be slow on first call. Cheap
  experiment to run before declaring `Kvaerno5` the final
  default: a side-by-side `Tsit5` dry-run on Stages 1–3. SOA
  exchange is *mildly* stiff (concentrations span many orders of
  magnitude — hence ADR-002's `float64` mandate — but the
  characteristic timescales aren't ratio'd hard apart like
  classical nucleation), so `Tsit5` may meet `rtol=1e-6` while
  amortizing compile cost better. Pick the default on data, not
  on compile-time anxiety. Document the head-to-head numbers in
  the PR description if `Tsit5` ends up the choice.
- **Are the 6 `xfail` markers in `tests/test_sweep.py` removed
  in this PR or in a follow-up?** Default: same PR. The xfail
  → pass flip is the load-bearing acceptance criterion.

## Risks

- Diffrax's choice of substeps at large `dt` (`nstep ∈ {1, 2, 4}`)
  could leave residuals above `rtol=1e-6` even at tight internal
  tolerances. If so, the diagnosis is itself useful — either it
  reveals a model formulation issue or it argues for tighter
  controller settings or a different solver.
- Compile-time regression on the operator-splitting driver. If
  the 60-step trajectory test goes from "tens of seconds" to
  "minutes," that's worth a separate decision on caching.
- `qgas_avg` divergence from Fortran's trapezoidal-over-adaptive-
  substeps formula — could shift the H₂SO₄ uptake feeding
  newnuc. Investigate empirically during validation.
