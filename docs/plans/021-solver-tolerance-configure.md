# Plan 021 — Process-global solver-tolerance configure() hook

**Status:** in progress (2026-06-13). PR open at [#58](https://github.com/reflective-org/MAM4-JAX/pull/58).
**Branch:** `feat/configurable-solver-tolerances` → `main`.
**Contributor:** @duncanwp (motivation: integrating MAM4-JAX into the jax-gcm GCM).

---

## 1. Scope

Add a process-global `solvers.configure()` hook so a host (e.g. a GCM driver) can dial the speed/accuracy/robustness trade-off of `solve_ivp` without threading kwargs through every call site. All-`None` arguments leave behavior unchanged — pre-existing tests stay byte-identical.

**Knobs**:
- `rtol` / `atol` — looser tolerances → far fewer adaptive PI-controller steps (the dominant per-call cost; the M3.6 / M7 call sites use tight defaults like `atol=1e-20` to match Fortran at machine ε).
- `max_steps` — per-call adaptive step cap.
- `throw` — `True` (upstream diffrax default) raises on `max_steps` exhaustion; `False` returns the best estimate with a non-success `sol.result` code instead. Lets a `vmap`ed host gate/log non-converging cells without aborting the batch.
- `reset` — clears all overrides back to the per-call `SolverConfig` defaults.

**Out of scope** (intentional, not omissions):
- `solver` choice (e.g. `Kvaerno5` vs `Tsit5`). Structural decision per call site (different stability regions, different memory cost). A host that wants a different solver should fork the relevant call site or vendor a new `SolverConfig`.
- `dt0` initial step hint. Same rationale — structural.

---

## 2. Motivation

@duncanwp profiled MAM4-JAX inside jax-gcm (T63L47, 866 k cells, A100):

| tolerances | amicphys speedup | max rel-err per 12-min step |
| --- | --- | --- |
| rtol=1e-9 / atol=1e-20 (current) | 1.0× | — (reference) |
| rtol=1e-6 / atol=1e-15 | **2.8×** | 0.13 % |
| rtol=1e-4 / atol=1e-12 | 3.0× | 2.3 % |

The per-cell `Kvaerno5` implicit solve dominates the per-step aerosol cost (~1.4 s/step at T63L47). Float32 gave no speed/memory benefit on the A100 — the solve is latency-bound, not bandwidth-bound — so the tolerance knob is the only meaningful lever. `throw=False` separately addresses the "one pathological cell aborts the batch" failure mode in `vmap`ed inference.

---

## 3. Design

### Override layering (in `solve_ivp`)

```
final_rtol = _OVERRIDE["rtol"] if _OVERRIDE["rtol"] is not None else config.rtol
final_atol = _OVERRIDE["atol"] if _OVERRIDE["atol"] is not None else config.atol
final_max_steps = _OVERRIDE["max_steps"] if _OVERRIDE["max_steps"] is not None else config.max_steps
final_throw = _OVERRIDE["throw"] if _OVERRIDE["throw"] is not None else True
```

- Falls back to per-call `SolverConfig` when override is `None`.
- `throw` defaults to `True` if neither override nor config sets it (matches upstream diffrax).

### JIT-cache contract

`solve_ivp` is called from inside `@jax.jit`-decorated functions (`_mam_amicphys_1subarea_clear`, `run_step`, etc.). The `_OVERRIDE[...]` reads happen at **trace time** — the chosen value is baked into the cached JIT binary. Reconfiguring after a path has been traced won't change the cached behavior; only a new trace (e.g. fresh process start, or different array shapes triggering a new specialization) picks up the new values.

**Pattern**: `solvers.configure(rtol=1e-6, atol=1e-15)` at process startup, before any `run_step(state)` call. Reconfiguring mid-run is supported semantically but takes effect only on uncompiled call sites.

### Thread safety

`_OVERRIDE` is a module-level mutable dict with no lock. Single-threaded use is safe. Process-per-device parallelism is safe (each process has its own `_OVERRIDE`). Single-process multi-threaded hosts calling `configure()` from different threads can observe non-deterministic interleaving — set once at startup or coordinate externally.

---

## 4. Tests (`tests/test_solvers.py`, 6 tests)

Uses a synthetic stiff scalar ODE (`dy/dt = -1000·y` over `[0, 0.01]`, analytical `y(t) = exp(-1000·t)`) — no MAM4 fixture dependency.

| Test | What it locks in |
| --- | --- |
| `test_configure_rtol_reduces_step_count` | `configure(rtol=1e-3, atol=1e-6)` produces fewer adaptive steps than the tight default (rtol=1e-9). Both still hit the analytical answer at the requested accuracy. |
| `test_configure_default_is_no_op` | `configure()` with all `None` leaves behavior byte-identical (no step-count change, no `ys` change). Safety property for hosts that import without calling `configure`. |
| `test_configure_throw_false_does_not_raise_on_step_exhaustion` | `configure(throw=False, max_steps=4)` returns without raising; `stats` surfaces the diagnostic. Load-bearing for `vmap`ed hosts. |
| `test_configure_throw_true_default_raises_on_step_exhaustion` | Upstream default behavior preserved — `max_steps=4` raises. Locks in against a regression to "silently swallow." |
| `test_configure_reset_clears_overrides` | `configure(reset=True)` falls back to per-call `SolverConfig` defaults. |
| `test_configure_reset_with_kwargs_applies_kwargs_after_reset` | `configure(reset=True, rtol=1e-6)` is "clear, then set rtol" semantic. |

`autouse` fixture resets overrides around every test so order doesn't matter and failures can't leak state.

---

## 5. Risks / known unknowns

1. **JIT cache surprise** (item flagged in #58 review). Documented in the `configure` docstring; tests can't easily verify trace-time semantics. A future contributor who skims the docstring and reconfigures mid-run will be confused. The mitigation is documentation, not code.
2. **Thread safety** (also flagged). Documented as a caveat. Real risk is low for the target jax-gcm use case (process-per-device).
3. **Per-call sites still set `atol=1e-20`** in places (e.g., M3.6 PR-D's H₂SO₄ solver). When a host loosens via `configure`, those tight call sites are loosened too. Empirically (jax-gcm's profile) this is the desired behavior — the GCM doesn't need bit-match-to-Fortran per step — but a future call site that genuinely needs the tight bar (e.g., a calibration workflow) would need a finer-grained API. Out of scope for now.

---

## 6. Open questions (none blocking)

1. **`solver` / `dt0` symmetry**: should the override include `solver` and `dt0` too? Decided **no** — structural choice, not runtime knob. Documented in `SolverConfig` docstring.
2. **Reproducibility of the speedup table**: ideally @duncanwp's profiling lands as `scripts/_benchmark_solver_tolerances.py` so the table can be re-measured. Not blocking the API itself but useful for empirical-claim provenance per CLAUDE.md rule #6. (TODO: ask @duncanwp.)

---

## 7. Pointers

- `mam4_jax/solvers.py` — the module.
- `tests/test_solvers.py` — the tests.
- PR [#58](https://github.com/reflective-org/MAM4-JAX/pull/58).
- ADR-013 (dual-branch), ADR-015 (3 % bar) — context for why the tight per-call defaults exist.
- jax-gcm integration (downstream consumer, external).
