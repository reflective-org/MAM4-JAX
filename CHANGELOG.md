# Changelog

## v0.4.0 — 2026-08-25

Completes the amicphys port: primary-carbon aging was the last sub-process out
of scope. It is **on by default**, which changes results for every caller on
defaults — see Changed.

### Added

- **Primary-carbon aging** (`mam_pcarbon_aging_1subarea`, Fortran
  `modal_aero_amicphys.F90:5111-5285`). Pcarbon particles that acquire a
  sulfate-equivalent hygroscopic shell thick enough to coat the mode transfer
  to the accumulation mode. Gated by `mdo_pcarbonaging` (default `1` — the
  Fortran has no toggle, so on is the faithful setting).

- **`configure_pcarbon_aging(n_so4_monolayers=...)`** — the coating threshold,
  in monolayers of so4. Default **3.0**, the value the amicphys path actually
  receives (`box_model_utils/phys_control.F90:26` ->
  `modal_aero_initialize_data.F90:417,585`). The frequently quoted `8.0`
  (`modal_aero_gasaerexch.F90:37`) belongs to the legacy `modal_aero_coag`
  aging path and is *not* what this code path uses.

- **`AmicphysParams`** — a pytree of traced, differentiable numeric knobs
  (`n_so4_monolayers`, `qgas_netprod_h2so4`, `qgas_netprod_soa`), passed via
  `amicphys`/`run_step`/`run_timesteps`'s `params` argument:

  ```python
  from mam4_jax import run_step, AmicphysParams
  run_step(state, AmicphysParams(n_so4_monolayers=3.0))
  ```

  Fields left `None` fall back to the `configure_*` process globals. Leaves are
  ordinary traced operands, so `jax.grad` reaches them and a parameter sweep
  reuses one compiled step instead of recompiling per value. Code-path
  selectors (the condensation `backend`, the `mdo_*` toggles) deliberately stay
  static — see ADR-020.

### Changed — behavioural

- **Pcarbon aging is on by default.** Against the Fortran box model this moves
  pcarbon tracers substantially — up to 14x on pcarbon BC and number over 60
  steps — because the previous behaviour was missing the process entirely.
  Callers who need the old results pass `mdo_pcarbonaging=0`.

- **Sulfur and SOA are now conserved across the pcarbon mode.** The pcarbon
  `LMAP_AER` row maps only pom/bc/mom, so so4/soa condensed or coagulated onto
  that mode had no `pcnst` slot and was silently dropped at the amicphys
  repack — a per-step sink. Aging moves those species to accumulation before
  the repack, which closes it.

- **Arithmetic deviations from the Fortran, both deliberate (ADR-019).**
  `1 - exp(-x)` is now `-expm1(-x)` at the coagulation transfer sites (~1e-10
  relative in f64, ~10 % in f32). `qs21`'s prefactor `(1+r6)**(2/3) - rx4` is
  now the algebraically exact `rx4 * expm1((2/3) * log1p(1/r6))`; the Fortran
  form cancels catastrophically and returns the wrong *sign* at large diameter
  ratios. `qs21` consequently differs from the Fortran reference by 1.19e-7
  where every other coefficient agrees to ~4e-16.

### Fixed

- **Float32 intermodal coagulation mass transfer was identically zero.**
  `getcoags` formed its harmonic means as `a*b/(a+b)`; the operands are
  representable in float32 but their product underflows min-normal, so
  `betaij3` — and with it *all* intermodal third-moment (mass) transfer —
  flushed to `0` in a float32 core. Number transfer was unaffected, so nothing
  looked broken. All seven harmonic means are now reciprocal-form. Four
  second-moment coefficients (`betaij2i`, `betaij2j`, `betaii2`, `betajj2`)
  were dead the same way and are also fixed; `v0.3.x`'s claim that only `qv12`
  was affected was wrong (see `docs/plans/023-*.md` §8).

  Float64 results are unchanged beyond the ADR-019 deviations above.

### Notes

- Requires no changes for hosts already on `0.3.x` beyond deciding whether they
  want aging on (they almost certainly do — without it, primary carbon never
  ages and BC lifetime runs roughly 3x observed).
- New ADRs: **ADR-019** (Fortran arithmetic deviations), **ADR-020** (numeric
  knobs as traced pytree leaves). Plan: `docs/plans/026-pcarbon-aging.md`.

## v0.3.2 — 2026-08-20

Same content as v0.3.0, which was never published. Two tags were burned getting
here, both for process reasons rather than anything wrong with the package:

- **`v0.3.0`** was cut before CI had ever run. The build failed in the test gate
  because Git LFS objects are not fetched by `actions/checkout` by default.
- **`v0.3.1`** was cut from a merge that contained only the first of the two
  fix commits, so it points at a tree whose `__version__` still says `0.3.0`.

The `tags` ruleset blocks deletion and non-fast-forward updates on all tags --
correctly; a tag someone may have fetched should not change underneath them --
so each attempt consumes a number rather than reusing one. Nothing was ever
published, so only dangling tags are left behind.

- CI now fetches Git LFS objects.
- CI now verifies the tag matches `mam4_jax.__version__` before building.
  Nothing previously tied them together.

## v0.3.0 — never published

First release intended for use as a library by another model, so the headline
change is that there is now a **public API**.

### Added

- **A stable top-level namespace.** Everything a host model needs is exported
  from `mam4_jax` directly:

  ```python
  from mam4_jax import run_step, run_timesteps, configure_gas_netprod
  ```

  Only names in `mam4_jax.__all__` are covered by semantic versioning.
  Submodule paths are internal and do move — see below.

- **`configure_gas_netprod(h2so4=..., soa=...)`** is now public. This is the
  hook a host model drives its own gas chemistry through: it sets the
  "other-process" gas production rate [mol/mol/s] that the aerosol system sees.
  It previously existed only at `mam4_jax.processes.amicphys`.

- `Topology` and the `(variant, nmodes)` configuration axis, with `nmodes = 5`
  (MAM5 `coarse_strat`) accepted and validated. **No CAM or MAM5 instance is
  registered yet** — the axis exists, the data does not.

- `topology_jit` / `trace_policy`. Reading the active topology inside a jitted
  function is now an error by default, because jit caches are not keyed on
  module globals and a kernel traced under one topology silently kept returning
  its answer after `set_topology`.

### Changed — breaking

- **The package was restructured.** `mam4_jax.processes.*` no longer exists:

  | was | now |
  | --- | --- |
  | `mam4_jax.constants`, `.data`, `.topology`, `.config` | `mam4_jax.core.*` |
  | `mam4_jax.coag`, `.newnuc`, `.kohler`, `.saturation` | `mam4_jax.physics.*` |
  | `mam4_jax.processes.calcsize`, `.wateruptake` | `mam4_jax.physics.*` |
  | `mam4_jax.processes.amicphys` | `mam4_jax.coupling.amicphys` |
  | `mam4_jax.solvers` | `mam4_jax.solver.solvers` |

  No compatibility shims: they are how one thing ends up with two names. Import
  from the top-level namespace instead and this cannot bite again.

- **Four dead modules deleted** — `processes/{coag,newnuc,gasaerexch,rename}.py`
  raised `NotImplementedError` and two of them *shadowed working
  implementations by name*, so `from mam4_jax.processes import coag` crashed
  while `from mam4_jax import coag` worked.

- The version is now single-sourced from `mam4_jax.__version__`.
  `pyproject.toml` said `0.0.1` while the newest tag was `v0.2.0-beta.1`.

### Fixed

- **`calcsize` enforced number bounds that the reference deliberately
  disables.** The `do_aitacc_transfer` branch that divides `v2nxx` and
  multiplies `v2nyy` by 1e6 for the aitken/accum modes was missing, so the port
  clamped number where CAM and E3SM do not (identical in both, so a shared-path
  defect).

- **The post-transfer `dgncur` recompute was missing entirely.** Restoring the
  bound turn-off above made the aitken↔accum transfer reachable for the first
  time, which exposed it. Without it the port diverged from the Fortran by
  9–32 % on `dgncur`, which feeds water uptake, coagulation and condensation.

  That recompute carries a known **bug in upstream CAM**: it uses a stale scalar
  `dumfac`, applying the wrong mode's width. We use the correct per-mode value;
  `bug_compat_stale_dumfac=True` reproduces CAM, which is needed to validate
  against the Fortran reference. See `mam-box-fortran` issue and
  `docs/bugs/BUG-cam-calcsize-stale-dumfac.md` there.

### Packaging

- Release is gated on the test suite and on installing the built wheel into a
  clean virtualenv, because the wheel has shipped without its `_coag_tables.npz`
  before (#62) and that file moved again in this release.
- Publishing uses PyPI Trusted Publishing — no API token is stored in the repo.
