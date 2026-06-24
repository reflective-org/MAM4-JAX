# Plan 023 — Float32-safe coag + `JAX_ENABLE_X64=0` opt-out

**Status:** in progress (2026-06-24). PR open at [#60](https://github.com/reflective-org/MAM4-JAX/pull/60).
**Branch:** `feat/coag-f32-safe` → `main`.
**Contributor:** @duncanwp (motivation: jax-gcm at T63L47 — float32 halves memory, doubles throughput on A100).

---

## 1. Scope

Two surgical changes that, together, let a host opt the entire coupled model into float32 when paired with the `"substep"` / `"astem"` condensation backends (plan 022):

1. **`mam4_jax/coag.py:getcoags`** — refactor the `qv12` (third-moment intermodal coagulation) coefficient from a direct harmonic mean of two `~1e-38` operands to a `dgat3`-factored harmonic mean of normal-range operands. Algebraically identical in float64; removes the float32 `0/0 = NaN` underflow.
2. **`mam4_jax/__init__.py`** — gate the unconditional `jax.config.update("jax_enable_x64", True)` behind the standard JAX env var `JAX_ENABLE_X64` (case-insensitive truthy values pass through; `"0"`, `"false"`, etc. leave x64 off).

Documented as ADR-018 (amends ADR-002).

---

## 2. What's in scope vs. deferred

| Item | This PR | Deferred to PR-NEXT |
| --- | --- | --- |
| qv12 algebraic refactor (coag third-moment) | ✅ | — |
| Auto-cast of `_BM*` lookup tables to caller dtype | ✅ | — |
| `JAX_ENABLE_X64` env-var honored at import | ✅ | — |
| `mam4_jax.x64_enabled` live PEP 562 read | ✅ | — |
| Import-time `UserWarning` listing f64-forced modules | ✅ | — |
| `test_getcoags_finite_in_float32` (full reference sweep) | ✅ | — |
| `test_jax_enable_x64_zero_opts_out` (subprocess) | ✅ | — |
| Gate `dtype=jnp.float64` casts in kohler/wateruptake/calcsize/amicphys/newnuc on live x64 state | — | ✅ (see §5) |
| Trajectory-level acceptance bar for f32 mode | — | ✅ (owner approval; ADR addendum) |
| Audit other coag coefficients (qs11/qs22/qs12/qs21) for f32 magnitude bounds | — | ✅ (only qv12 is f32-broken; the qs* are below f32 useful range but were already so before this PR) |

---

## 3. Why qv12 specifically

`qv12 = coagnc3 · coagfm3 / (coagnc3 + coagfm3)` is the harmonic mean of two coefficients, both `∝ dgat3 = dgatk³`. In SI metres, `dgatk ~ 3e-8` so `dgat3 ~ 1e-23`. Multiplied by the two Whitby-series prefactors, both `coagnc3` and `coagfm3` land at `~1e-38`. Float32's smallest normal is `~1.18e-38` — both terms underflow to 0 and the ratio becomes `0 / 0 = NaN`.

The fix factors `dgat3` out symmetrically:

```
coagnc3 = dgat3 · nc3,   coagfm3 = dgat3 · fm3
qv12    = dgat3 · (nc3 · fm3) / (nc3 + fm3)
```

`nc3` and `fm3` are in normal range (~1e-7 and ~1e-13), the harmonic mean is well-defined, and only the final `dgat3 · …` multiply is tiny. Algebraically identical in float64 to round-off; the float64 reference is unchanged (`test_getcoags_matches_fortran` passes at `rtol=1e-6` with the new expression). In float32, `qv12` is now finite at rel-err ~5.9e-8.

The qv12 site is unique in `getcoags`: every other harmonic-mean site (`i1`, the second-moment intermodal) uses `dgat2` or `sqdgat5`, which stay in normal float32 range.

## 4. Why the env-var gate

Without §1 the gate is useless — `getcoags` returns NaN regardless. With §1 in place, the `"substep"` / `"astem"` condensation backends (plan 022) are float32-safe end-to-end, but the package still hard-enables x64 on import. The gate lets a host opt out via JAX's standard `JAX_ENABLE_X64=0` env var (set before import).

We honor JAX's own truthy spelling (`1`/`true`/`yes`/`on`, case-insensitive) so users who set `JAX_ENABLE_X64=false` get the behaviour they expect.

When the opt-out is in effect, we emit a `UserWarning` at import:

- Naming the modules with explicit `dtype=jnp.float64` casts (those casts silently downcast to float32 + emit JAX promotion warnings).
- Naming the `"diffrax"` backend as numerically unsafe in float32 (atol=1e-20).
- Pointing to ADR-018.

## 5. Known limitations (deferred to a follow-up PR)

- `kohler.py`, `processes/wateruptake.py`, `processes/calcsize.py`, `processes/amicphys.py`, `processes/newnuc.py` contain ~25 explicit `dtype=jnp.float64` casts. With `JAX_ENABLE_X64=0` those casts emit JAX `UserWarning` and silently downcast to float32. A follow-up PR must gate each on live `mam4_jax.x64_enabled` (or equivalent) before the f32 path is truly clean. Currently the f32 path WORKS — JAX downcasts on each cast — but it's noisy and not validated end-to-end.
- No trajectory-level acceptance bar for the f32 coupled run. ADR-018 §5 captures this as deferred (ADR-015's 3 % / 24 h is the natural starting point pending owner approval).
- In-process `jax.config.update("jax_enable_x64", False)` after import is only partially supported: `getcoags` defensively `.astype()`s its lookup tables; other modules MAY emit promotion warnings on toggle.

## 6. Files touched

```
mam4_jax/__init__.py           (~50 lines: env-var gate, UserWarning, PEP-562 __getattr__)
mam4_jax/coag.py               (~25 lines: qv12 refactor + lookup-table .astype)
tests/test_coag.py             (~90 lines: f32 finiteness sweep, subprocess opt-out test)
tests/test_scaffolding.py      (~5 lines: skip-when-opted-out gating)
tests/* + scripts/*            (sed: standardize `# noqa: F401` comment trailers)
docs/KEY_DECISIONS.md          (ADR-018)
docs/plans/023-…md             (this plan)
docs/FEATURES.md               (1 row: float32-safe getcoags qv12)
docs/PROGRESS.md               (entry)
```

## 7. Acceptance

- `pytest tests/test_coag.py -v` — 4/4 pass (default x64=on).
- `python3 -W error::UserWarning -m pytest tests/test_coag.py::test_getcoags_finite_in_float32` — passes (no JAX promotion warnings).
- `JAX_ENABLE_X64=0 python -c "import mam4_jax; import jax; assert not jax.config.read('jax_enable_x64'); assert mam4_jax.x64_enabled is False"` — passes.
- Existing test suite (under default x64=on) unchanged.
