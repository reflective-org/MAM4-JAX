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

Replace the H₂SO₄ analytical block in
`_mam_gasaerexch_1subarea` (the `tmp_kxt` 3-branch +
mode-distribution code path, currently `Stage B` and `Stage C`
comment-banners inside that function) with a `solve_ivp` call.

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
`atol = 1e-20` is small enough relative to typical `h2so4_gas`
magnitudes (~1e-12 mol/mol in the box fixture, sometimes smaller)
that the controller is in the rel-tol regime where it matters
(`atol + rtol·|y|` → `rtol·|y|` dominates for |y| > 1e-11 or so).
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
3 %; the worst field's rel-err is still expected to be dominated
by `soag_gas` at ~2.55 % (PR-D2 doesn't touch soaexch).

**Per-field acceptance target for `h2so4_gas`:**

- **Hard floor (blocks the PR):** ≤ 0.5 % at dt=5s. The current
  baseline on `diffrax` is 0.31 %; PR-D2 must not regress by more
  than ~1.6×. A regression beyond that would indicate the diffrax
  port is *worse* than the handwritten 3-branch closed form on its
  own ODE, which would be surprising and worth investigating
  before merging.
- **Stretch target:** ≤ 0.1 % at dt=5s. The "same ODE → tight
  match" reasoning (see Open Questions) predicts an
  order-of-magnitude improvement. If we hit this, the qgas_avg
  endpoint trapezoidal is empirically benign on the H₂SO₄ side.
- **Diagnostic dt=30 / dt=300:** should not degrade relative to
  current `diffrax` baseline (3.13e-3 / 3.51e-3 respectively).
  Same scaling story as `soag_gas` — coarse-dt observational
  only.

### Plots

Re-run `scripts/diffrax_24h_plot.py` after the port; the per-mode
`traj_*_24h_dt*.png` figures auto-regenerate. The `h2so4_gas`
panel in `traj_gas_24h_dt*.png` is the visual deliverable showing
PR-D2's effect.

## Open questions

- **`qgas_avg[igas_h2so4]` integration strategy.** Default:
  `(g_prv + g_new) / 2` endpoint trapezoidal. Fallback if newnuc
  validation diverges: switch to a denser `SaveAt` grid (e.g.
  4–8 points across the substep) and trapezoidal-integrate, OR
  compute the exact integral analytically from the recorded
  endpoints (`((g_prv - q_eq)·(1-e)/(tmpa) + q_eq·dt) / dt`) —
  but that reintroduces the 3-branch numerical guard, partially
  defeating the migration.

  **Concrete decision criterion** (so the choice is reproducible,
  not vibes): after the initial endpoint-trapezoidal port, look
  at `h2so4_gas` rel-err vs dt across the 4-dt sweep.

  - **If rel-err is roughly dt-independent** (PR-D1 soag_gas
    signature: ~constant 2.4 % across dt), the trapezoidal is
    leaving a structural offset. Switch to denser `SaveAt`.
  - **If rel-err shrinks with finer dt** (order ~`dt`), the
    trapezoidal endpoint converges as expected and is fine to
    keep — the offset is solver-truncation, not formula choice.
  - **Only consider the analytical-integral fallback** if denser
    `SaveAt` also doesn't close the gap *and* the structural
    offset blocks the acceptance target. It's a last resort
    because of the migration-defeating 3-branch logic.
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
  same answer to ~ε *for the H₂SO₄ ODE alone*. So **no structural
  offset is expected from the H₂SO₄ solver itself.**

  **Caveat:** H₂SO₄ doesn't live in isolation.
  `qgas_avg[igas_h2so4]` feeds newnuc → newnuc mutates mode
  populations → mode populations change `uptkaer_h2so4` for the
  next outer driver step. If our endpoint-trapezoidal
  `qgas_avg` differs measurably from Fortran's exact analytical
  `q4` (it will — trapezoidal-on-two-points biases high for a
  convex `exp(-t)`), newnuc consumes slightly different input,
  and the cumulative trajectory drift on `h2so4_gas` over 24 h
  is set by **that feedback loop**, not by H₂SO₄ solver
  truncation.

  Realistic prediction (revised from earlier draft): rel-err
  drops from the current 0.31 % by at least an order of
  magnitude — somewhere in the ~1e-3 to ~1e-5 range, not to
  ε. **If it doesn't drop, suspect qgas_avg-newnuc coupling
  rather than the H₂SO₄ port itself** — the §1 fallback
  chain above is the right next move.

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
  dt (1s), this overhead compounds across many calls.

  **Threshold and fallback (with teeth):** if the 24 h sweep at
  dt=1s slows by more than ~2× over the current runtime
  (post-PR-D1 baseline), the explicit decision on the table is
  **abort PR-D2 and keep the analytical 3-branch block in
  place** until M6 closes the JIT-boundary gap. Reverting is
  the actual "fallback" — the diffrax migration's value
  proposition rests on physical-accuracy improvement, and if
  that improvement comes at a 2×+ runtime cost on a path where
  the handwritten code already produces the true ODE solution,
  the trade isn't worth it.

## What this PR does NOT do

- **No new fixtures.** Reuses PR-D1's 24h infrastructure.
- **No M6 / operator-splitting work.** Bar relaxation from PR-D1
  stays.
- **No `_mam_coag_1subarea` port** — that's PR-D3, deferred
  unless motivated.
- **No `qgas_netprod` rework** — stays hardcoded at 1e-16.
- **No `jit` boundary tightening** — Phase A discipline.
