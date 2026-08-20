# Changelog

## v0.3.0 — unreleased

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
