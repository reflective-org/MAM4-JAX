# Plan 022 — Operator-split condensation backends for gasaerexch

**Status:** in progress (2026-06-13). PR open at [#59](https://github.com/reflective-org/MAM4-JAX/pull/59).
**Branch:** `feat/substep-condensation` → `main`.
**Contributor:** @duncanwp (motivation: integrating MAM4-JAX into the jax-gcm GCM at T63L47).

---

## 1. Scope

Add two **opt-in** condensation backends inside `_mam_gasaerexch_1subarea`, selectable via a new process-global hook:

```python
import mam4_jax.processes.amicphys as amicphys
amicphys.configure_condensation(backend="substep", n_substeps=4)  # fast operator-split
amicphys.configure_condensation(backend="astem")                  # Fortran-faithful adaptive
```

Default stays `"diffrax"` — pre-existing behaviour is unchanged unless a host explicitly opts in. Mirrors the `solvers.configure` pattern from PR #58 (plan 021).

The two new backends replace the adaptive Kvaerno5 solve on both the H₂SO₄ uptake ODE (linear in `g`) and the SOA exchange ODE (nonlinear only through `g_star(a)`):

- **`"substep"`** — analytic exact closed-form H₂SO₄ + `n_substeps` fixed-step SOA with `g_star` frozen per substep (each substep is then a linear closed form). `lax.scan`, no per-cell loop. Fastest; reverse-mode differentiable.
- **`"astem"`** — analytic exact H₂SO₄ + a port of upstream `mam_soaexch_1subarea`'s adaptive semi-implicit step1/step2 Euler loop with substep `dtcur = alpha_astem / tmpa`. `jax.lax.while_loop`, capped at `_NITER_MAX_ASTEM = 1000`. Matches the CAM/E3SM discretization exactly. **Not reverse-mode differentiable** (`lax.while_loop` is forward-only).

---

## 2. Trade-off matrix

| backend | scheme | speed | per-call rel-err vs tight Kvaerno5 | autodiff (`jax.grad`) | adaptive |
| --- | --- | --- | --- | --- | --- |
| `"diffrax"` (default) | Kvaerno5 + PIDController | 1.0× (tight) / ~2.8× (rtol=1e-6) | 0.13 % at rtol=1e-6 (PR-58 measurement) | ✓ (PR-J5 audited) | yes |
| `"substep"` | analytic H₂SO₄ + frozen-`g_star` substep (`lax.scan`) | ~55× at n=4 (PR-59 T63L47) | 0.28 % | ✓ (`lax.scan` is grad-OK) | no (fixed N) |
| `"astem"` | analytic H₂SO₄ + Fortran semi-implicit Euler (`lax.while_loop`) | ~38× | 1.17 % (CAM/E3SM's own 1st-order discretization error) | ✗ (`lax.while_loop` forward-only) | yes |

`n_substeps` barely affects `"substep"` speed — pick it for accuracy.

Multi-day validation (ECHAM + JAM-MAM4 T21, 3 sim-days): substep and astem agree to **0.18 % on the 3-day global aerosol burden**. Codified at per-call level by `test_substep_and_astem_agree_per_call`.

---

## 3. `qgas_avg` semantics differ between backends

This is a subtle behavioural difference worth documenting up-front so a future debugger doesn't lose time chasing it.

The `qgas_avg` field accumulates the **time-averaged gas vmr** over the substep — consumed downstream by newnuc (M3.6 PR-F3). The three backends produce qualitatively different `qgas_avg`:

| backend | qgas_avg integration |
| --- | --- |
| `"diffrax"` | Endpoint trapezoidal of the diffrax-solved trajectory: `(g(0) + g(t1)) / 2`. Approximate. |
| `"substep"` / `"astem"` | **Exact time-mean of the closed-form g(t)** over each substep: `∫₀ᵈᵗ g(t) dt / dt = (g(0) − g(dt)) / (K · dt) + src/K` for the linear-in-g substep. Exact. |

The substep/astem `qgas_avg` is mathematically better (the exact time-mean of the analytic solution vs the trapezoidal-of-endpoints approximation). It's the same quantity the Fortran reference is computing. For host workflows that consume `qgas_avg` (newnuc), the substep/astem path is strictly more faithful — no accuracy regression.

Plan-017 (diffrax-h2so4) introduced the endpoint-trapezoidal convention for the diffrax path. That plan is closed; the difference is captured here.

---

## 4. Autodiff implications

PR-J5 (M6) audited the codebase as reverse-mode-differentiable: `jax.grad` through `run_step` and 60-step `scan` returns finite, deterministic cotangents. That audit covered the `"diffrax"` backend.

| backend | reverse-mode diff | path |
| --- | --- | --- |
| `"diffrax"` | ✓ | diffrax `RecursiveCheckpointAdjoint` (PR-J5) |
| `"substep"` | ✓ | `lax.scan` is reverse-mode-diff |
| `"astem"` | ✗ | `lax.while_loop` is forward-mode-only — `jax.grad` raises |

The astem ✗ is locked in by `test_astem_backend_not_grad_compatible`: a future "fix" that silently swaps the while-loop for a grad-compatible construct (e.g., `lax.fori_loop` with a static cap) would change the per-cell adaptive behavior and should be a deliberate review, not a drive-by.

**For M9 calibration workflows**: use `"diffrax"` or `"substep"`. `"substep"` is faster; `"diffrax"` is the audited default.

---

## 5. Validation bar

Per-call test bar is `rtol=1e-2, atol=1e-12` on `q`/`qqcw` (the diffrax-branch convention). This is a per-call equivalence check against the Fortran reference (and, for the cross-validation test, between substep and astem).

The relationship to ADR-015's trajectory bar (3 % / 24 h / dt ≤ 5 s) is formalized in **ADR-017** ("Per-call equivalence bar for opt-in solver backends") landing alongside this PR. The short version: opt-in backends pass per-call equivalence at `rtol=1e-2` and trajectory equivalence at ADR-015's 3 %; hosts that need tighter bars run the default `"diffrax"` backend.

---

## 6. Tests (`tests/test_amicphys.py`, 5 condensation tests total)

| Test | What it locks in |
| --- | --- |
| `test_condensation_backend_default_is_diffrax` | Opt-in contract — nothing changes unless `configure_condensation` is called. |
| `test_condensation_substep_matches_fortran` | substep backend reproduces Fortran gasaerexch+soaexch at `rtol=1e-2`. |
| `test_condensation_astem_matches_fortran` | astem backend reproduces the same fixture at `rtol=1e-2`. |
| `test_substep_and_astem_agree_per_call` | Cross-validation — substep and astem agree at `rtol=1e-2`. Catches a regression in either that the Fortran-match tests wouldn't surface alone. |
| `test_astem_backend_not_grad_compatible` | Documented contract: `jax.grad` through astem raises (per `lax.while_loop` constraint). |

Each test saves+restores `_COND` in a `try/finally` so process-global state doesn't leak between tests.

---

## 7. Risks / known unknowns

1. **`_NITER_MAX_ASTEM = 1000` silent unconverged**: under `vmap`, the batched `while_loop` is paced by the stiffest cell. If a cell exhausts the cap, the loop exits **silently** with that cell unconverged — no error raised. Host's responsibility to gate via finite-check downstream. Documented in `configure_condensation`'s docstring.
2. **JIT cache contract**: `configure_condensation`'s effects are baked into the JIT trace. Reconfiguring after a path has been traced has no effect on the cached binary. Documented in the docstring (same pattern as `solvers.configure`).
3. **Thread safety**: `_COND` is a module-level mutable dict with no locking. Set once at startup or coordinate externally.

---

## 8. Open questions

1. **Mutable-dict-per-module config pattern is starting to spread** (PR #58 added `_OVERRIDE` in `solvers.py`; this PR adds `_COND` in `amicphys.py`). If a future PR adds backend-toggles for newnuc / coag, that's 4 mutable module-level dicts. **Forward-looking concern**, not a blocker: when the pattern hits a third module, revisit whether to centralize the runtime-config layer (e.g., a single `mam4_jax.config` module with a `configure(component, **kwargs)` API) or keep it per-module. No action this PR; flagged for the next runtime-config addition.
2. **`alpha_astem` and `niter_max` from Fortran source**: bit-fidelity with a specific CAM/E3SM build requires confirming these constants match. Currently `0.05` and `1000` respectively, per the upstream source vendored in `mam4-original-src-code/`. If a host targets a different CAM version, these may need adjusting.

---

## 9. Pointers

- `mam4_jax/processes/amicphys.py` — `configure_condensation`, `_linear_uptake_closed_form`, `_soaexch_substep`, `_soaexch_astem`.
- `tests/test_amicphys.py` — five condensation tests.
- PR [#59](https://github.com/reflective-org/MAM4-JAX/pull/59).
- ADR-013 (dual-branch), ADR-015 (3 % bar), **ADR-017** (per-call equivalence bar — new this PR).
- Plan 017 (diffrax-h2so4) — context for the diffrax path being replaced.
- Plan 021 (`solvers.configure`) — sibling pattern.
