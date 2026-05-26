# Plan 017 — M7 PR-D2: port the H₂SO₄ analytical solver to diffrax

> **Status:** proposed 2026-05-25. Awaiting owner approval before
> implementation.

---

## Context

PR-D1 (PR #34, merged 2026-05-25) ported `_mam_soaexch_1subarea` to
diffrax, established the 24h validation framework, and revised
ADR-015's acceptance bar to **<3 % over 24 h at dt ≤ 5 s**. PR-D2
ports the second of the three handwritten solvers identified in
`docs/HANDWRITTEN_SOLVER_LIMITATIONS.md`: the **H₂SO₄ analytical
uptake** inside `_mam_gasaerexch_1subarea`.

The H₂SO₄ block is structurally different from soaexch:

- It's an **analytical closed-form** (3-branch formula on `tmp_kxt`,
  see `mam4_jax/processes/amicphys.py:587-649`), not a semi-implicit
  iteration. The closed form is essentially exact for the ODE it
  solves; the 3 branches are numerical-precision guards
  (`tmp_kxt > 0.001` exponential, `<= 0.001` Taylor, `< 1e-20`
  uptake-negligible).
- It includes a **non-zero gas-chem source term**
  (`qgas_netprod_h2so4 = 1e-16 mol/mol/s`, driver.F90:1248) — total
  sulfate mass grows over time.
- The per-mode aerosol gain `da[i]/dt = uptkaer_h2so4[i] · g` is
  linear in `g` with no nonlinear coupling (unlike soaexch's
  `g_star(a)`).

This makes the ODE simpler than soaexch's and the diffrax port is
expected to match Fortran much more tightly than PR-D1 did.

## The ODE

State vector per (col, level): `y = [g, a[0], a[1], a[2], a[3]]` of
length `NTOT_AMODE + 1`. With Fortran convention `igas_h2so4 = 1`,
`iaer_h2so4 = 1`, `tmpa = sum(uptkaer_h2so4)`,
`qgas_netprod = 1e-16`:

```
dy[0]/dt = -tmpa · y[0] + qgas_netprod
dy[1+i]/dt = uptkaer_h2so4[i] · y[0]    for i in 0..NTOT_AMODE-1
```

Closed-form (what the current handwritten port computes directly):

```
q_eq    = qgas_netprod / tmpa            # steady-state gas
g(t)    = (g0 - q_eq) · exp(-tmpa·t) + q_eq
a_i(t)  = a_i(0) + uptkaer_i · ∫₀ᵗ g(s) ds
        = a_i(0) + uptkaer_i · ((g0 - q_eq) · (1 - exp(-tmpa·t)) / tmpa + q_eq·t)
```

The handwritten port computes `g(dt)` as `q3` (3-branch) and
`(1/dt) · ∫₀^dt g(s) ds` as `q4` (also 3-branch); the per-mode
aerosol gain is `tmpa·dt · q4` distributed proportional to
`uptkaer_h2so4[i]/tmpa`.

## Scope (this PR, `diffrax` branch only)

### Implementation

Replace the analytical block in `_mam_gasaerexch_1subarea`
(lines 587-649) with a `solve_ivp` call.

**Proposed RHS** (sketch):

```python
def _h2so4_rhs(t, y, args):
    """RHS of H2SO4 gas/aerosol uptake ODE.

    y[..., 0]     = g (H2SO4 gas)
    y[..., 1:]    = a[mode] (sulfate aerosol per mode)

    args = (tmpa, uptkaer_per_mode, qgas_netprod)
    """
    tmpa, uptk, src = args
    g = y[..., 0]
    flux = uptk * g[..., None]            # (..., NTOT_AMODE)
    dg = -jnp.sum(flux, axis=-1) + src    # total uptake + source
    return jnp.concatenate([dg[..., None], flux], axis=-1)
```

**SaveAt strategy.** Use `SaveAt(t0=True, t1=True)` so we can
compute `tmp_q4 = (g0 + g_end) / 2` (the trapezoidal time-average,
which newnuc reads via `qgas_avg[igas_h2so4]`). The Fortran's
closed-form `q4` is exact for the underlying ODE; trapezoidal
endpoint over the diffrax trajectory is a different approximation
to the same integral. **The qgas_avg formula choice may matter
here** more than it did in PR-D1 because:

- Unlike soaexch's `qgas_avg[0]` which had no consumer, `qgas_avg[
  igas_h2so4]` IS consumed by newnuc (PR-D1's qgas_avg trace
  confirmed this).
- For `tmp_kxt > 0.001` the exact `q4` integral and the endpoint
  trapezoidal can differ measurably; for fine dt where
  `tmp_kxt « 1`, they converge.

If endpoint trapezoidal causes a structural offset on h2so4_gas
similar to PR-D1's soag_gas pattern, switch to a denser `SaveAt`
or compute `q4` directly using diffrax-recorded states. Decide
empirically during validation.

**Skip-mode handling.** Same as PR-D1: rely on
`uptkaer_h2so4[i] = 0` for skipped modes producing zero flux
naturally, no `where` chain in the RHS. The Fortran applies a
guard at `tmp_kxt < 1e-20` (uptake essentially zero, no qaer
update); diffrax with a non-negative tmpa will integrate
trivially through that regime and produce the same result.

**Caller-side cleanup.** The 3-branch `tmp_kxt` logic, the
`safe_kxt` divisor guards, and the `use_A`/`use_B`/`use_C` masks
are deleted along with the analytical block. The mode-distribution
math (`frac_per_mode = uptkaer/tmpa`) is also removed — diffrax
distributes the gas-phase loss across modes automatically through
the RHS.

**Solver configuration.** Per-call `SolverConfig(rtol=1e-9,
atol=1e-20)` to match PR-D1's choices. Kvaerno5 + PIDController.
The H₂SO₄ ODE is **less stiff** than SOA's because there's no
nonlinear coupling — Tsit5 (explicit RK4(5)) might compile faster
and match the same accuracy. Dry-run both during implementation
and pick on data (same "pick on data, not compile-time anxiety"
framing as PR-D1).

### Tests

Reuse PR-D1's 24 h validation framework wholesale. The
`tests/test_sweep.py` 4-dt parametrization (1s/5s/30s/300s at the
3 % bar for fine dt, diagnostic for coarse dt) already exercises
the H₂SO₄ analytical path inside `_mam_gasaerexch_1subarea` —
PR-D2's change will show up as a delta on `h2so4_gas` and
`so4_aer` per-mode rel-err in the same test.

No new fixture needed. The existing
`tests/reference/sweep_24h_no_pcarbon_aging/` references the box
model with all `mdo_*` toggles on; both SOA and H₂SO₄ paths
exercise.

**Acceptance:** `tests/test_sweep.py[1|5]` continues to pass at
3 %; ideally the worst field's rel-err **improves** (currently
~2.55 % dominated by soag_gas, with h2so4_gas at 0.31 % at dt=5).
PR-D2 shouldn't shift soag_gas (no soaexch change); h2so4_gas may
shift modestly (up or down). Diagnostic dt=30 / dt=300 cases
similarly should not degrade.

### Plots

Re-run `scripts/diffrax_24h_plot.py` after the port; the per-mode
`traj_*_24h_dt*.png` figures auto-regenerate. The `h2so4_gas`
panel in `traj_gas_24h_dt*.png` is the visual deliverable showing
PR-D2's effect.

## Open questions

- **`qgas_avg[igas_h2so4]` integration strategy.** Default:
  `(g_prv + g_new) / 2` endpoint trapezoidal. Fallback if newnuc
  validation diverges: switch to a denser `SaveAt` grid and
  trapezoidal-integrate, OR compute the exact integral
  analytically from the recorded endpoints
  (`((g_prv - q_eq)·(1-e)/(tmpa) + q_eq·dt) / dt`) — but that
  reintroduces the 3-branch numerical guard, partially defeating
  the migration.
- **Solver choice: Kvaerno5 vs Tsit5.** Kvaerno5 is the PR-D1
  default; Tsit5 is explicit RK and may compile faster on a
  not-very-stiff ODE like this one. Pick on validation data, not
  anxiety. Document the choice in the PR description.
- **`qgas_netprod_h2so4` as a constant.** Currently hardcoded to
  `1e-16` (matching `driver.F90:1248`). PR-D2 doesn't change this;
  if M6 ever wants to make gas-chem a real process, this constant
  becomes a state-dependent term — out of scope here.
- **Will PR-D2 expose a structural offset like PR-D1's?** PR-D1's
  soag_gas saturated at 2.4 % across dt because diffrax-true-ODE
  vs Fortran-semi-implicit have inherently different per-step
  outputs. For H₂SO₄, both Fortran and (proposed) diffrax solve
  the **same exact ODE** — Fortran's "analytical" is the true
  solution; diffrax's adaptive Kvaerno5 should converge to the
  same answer to ~ε. So **no structural offset is expected**.
  Cumulative trajectory difference should be at solver-truncation
  levels (~1e-9 ish at `rtol=1e-9`), and h2so4_gas rel-err vs
  Fortran should DROP from the current 0.31 % toward machine
  precision. This prediction is testable; if violated, it points
  to a bug in the port, not a fundamental issue.

## Risks

- **The qgas_avg integration story.** Newnuc consumes
  `qgas_avg[igas_h2so4]`; if our trapezoidal endpoint differs
  measurably from Fortran's exact `q4`, newnuc sees different
  input and the validation may shift in unexpected ways.
  Mitigation: validate first with endpoint trapezoidal; if
  insufficient, fall back to a denser SaveAt.
- **JIT compile time.** Each new `solve_ivp` call site introduces
  another JIT trace. The driver currently has soaexch + (after
  PR-D2) H₂SO₄ in diffrax; both compile separately on first call.
  The M4 60-step test runtime should be re-checked after PR-D2 to
  catch any regression.
- **Per-call overhead.** The H₂SO₄ closed-form is currently a
  handful of jnp ops; the diffrax call has more overhead (term
  construction, controller state, step decisions). At fine outer
  dt (1s), this overhead compounds across many calls. If the
  24h sweep at dt=1s slows by more than ~2× over the current
  runtime, that's worth a decision on caching or fallback.

## What this PR does NOT do

- **No new fixtures.** Reuses PR-D1's 24h infrastructure.
- **No M6 / operator-splitting work.** Bar relaxation from PR-D1
  stays.
- **No `_mam_coag_1subarea` port** — that's PR-D3, deferred
  unless motivated.
- **No `qgas_netprod` rework** — stays hardcoded at 1e-16.
- **No `jit` boundary tightening** — Phase A discipline.
