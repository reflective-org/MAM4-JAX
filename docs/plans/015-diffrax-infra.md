# Plan 015 — M7 PR-I1: diffrax migration infra & tooling

> **Status:** approved 2026-05-22. Ready to start.

---

## Context

ADR-013 established a dual-branch strategy: `main` keeps handwritten
solvers; a parallel `diffrax` branch ports them to
[`diffrax`](https://github.com/patrick-kidger/diffrax)-based
equivalents. ADR-014 (introduced by this PR) updates ADR-013 in two
ways:

1. The eventual `diffrax → main` merge **is** planned — `diffrax` is
   the canonical-to-be implementation, not a parallel experiment.
2. Cross-branch sync uses periodic `main → diffrax` **merges** (not
   cherry-picks, not rebases), preserving history for the merge-back.

PR-I1 is the first PR on the `diffrax` branch. It does **no solver
swap**. Its job is to establish the infrastructure all later solver
PRs need: the abstraction layer, the dependency, the baseline tag,
the limitation doc, and the new ADR.

## Scope

### On `main` (one-time, at merge of this PR or the PR-I1 PR)

- **Annotated tag `v0.1.0`** at the `main` tip immediately before any
  `diffrax`-branch work merges back. Tag message documents the
  handwritten-solver baseline: M3.6 amicphys complete, M4 driver,
  M5 partial (6 of 12 sweep step counts validated; 6 xfailed).
  Command (for reference, not run by this plan): `git tag -a v0.1.0`
  pointing at the appropriate commit.

### On `diffrax` branch

- **`pyproject.toml`**: add `diffrax >= X.Y` to dependencies (pin a
  current minor version; revisit in PR-D1 if the chosen solver needs
  a newer release).
- **`mam4_jax/solvers.py`** — new module, strategy-pattern wrapper
  around diffrax. Skeleton only — no real solver bodies wired in yet.
  Proposed signature:

  ```python
  from dataclasses import dataclass
  from typing import Callable, Optional
  import diffrax
  import jax.numpy as jnp

  @dataclass(frozen=True)
  class SolverConfig:
      """Per-call solver configuration consumed by `solve_ivp`."""
      solver: str = "Kvaerno5"        # diffrax solver class name
      rtol: float = 1e-9              # adaptive controller rel-tol
      atol: float = 1e-12             # adaptive controller abs-tol
      max_steps: int = 4096
      dt0: Optional[float] = None     # initial step (None = auto)

  @dataclass(frozen=True)
  class SolverResult:
      """Standardized return from `solve_ivp`."""
      ts: jnp.ndarray                 # recorded times (>= 1 entry)
      ys: jnp.ndarray                 # recorded states, leading axis = ts
      stats: dict                     # diffrax solver stats:
                                      #   num_steps, num_accepted_steps,
                                      #   num_rejected_steps, max_steps

  def solve_ivp(
      rhs: Callable,                          # (t, y, args) -> dy/dt
      y0,                                     # initial state (jax array / pytree)
      t0: float,
      t1: float,
      args=None,
      saveat: Optional[diffrax.SaveAt] = None,  # default: endpoint only
      config: SolverConfig = SolverConfig(),
  ) -> SolverResult:
      """Integrate dy/dt = rhs(t, y, args) from t0 to t1.

      Default `saveat=None` records `t1` only — endpoint-fast path for
      call sites that need just the terminal state (read via
      `result.ys[-1]`). Pass `diffrax.SaveAt(ts=...)` to record a
      trajectory at a chosen grid. `stats` always carries diffrax's
      step counts so adaptive-controller behavior is observable
      without per-call-site instrumentation.

      JIT-traceable in `y0` / `args`; not in `config` or `saveat`.
      """
      raise NotImplementedError("PR-I1 skeleton; wired up in PR-D1.")
  ```

  **Why this shape (not narrower).** A pure
  `(rhs, y0, t0, t1) -> y_final` wrapper would barely justify the
  indirection — it forwards three arguments and discards diffrax's
  diagnostics. Three reasons the wider shape is load-bearing:

  1. **Trajectory recording is required for validation.** Per the
     project's validation discipline, we compare JAX-driven
     trajectories against Fortran, not endpoints. PR-D1's
     soaexch port needs to record the solver's chosen substep
     trajectory so it can be plotted against Fortran's adaptive
     substeps. `SaveAt(ts=...)` is the diffrax mechanism; the
     wrapper must surface it.
  2. **Diagnostics are needed to tune the adaptive controller.**
     Step counts (accepted / rejected / total) tell us whether
     Kvaerno5 at `rtol=1e-9` is doing reasonable work or fighting
     stiffness. Hiding `solver_state.stats` forces every PR to
     re-instrument by hand.
  3. **The abstraction earns its keep this way.** Standardizing
     `(ts, ys, stats)` across all three solver call sites
     (soaexch, H₂SO₄, coag-if-needed) is the actual value of
     `solvers.py`; without it, the wrapper is cosmetic.

  Endpoint-only consumers stay simple (`solve_ivp(...).ys[-1]`).
  No per-call-site `saveat` boilerplate is forced. Reviewer
  should still push back if any field looks speculative.

- **`docs/KEY_DECISIONS.md` — append ADR-014**: dual-branch strategy
  reaffirmed; sync via merge; eventual merge-back planned. Annotates
  ADR-013's status as "Partially superseded by ADR-014" without
  editing its body (KEY_DECISIONS.md convention: never edit an
  Accepted ADR).
- **`docs/HANDWRITTEN_SOLVER_LIMITATIONS.md`** — new doc, findable
  from the `v0.1.0` tag's release notes. Enumerates:
  - Which solver call sites are handwritten (the three listed in
    M7's "Solvers in scope" bullet of PLANS.md).
  - The known M5 `nstep ≤ 30` gap, with the worst-case rel-err
    quoted from `docs/plans/014-convergence-sweep.md`.
  - Pointer to ADR-013, ADR-014, and the `diffrax` branch as the
    resolution path.
- **`docs/PLANS.md` M7 section** — already updated in the planning
  PR that archives this file. PR-I1 itself doesn't need to touch it
  again unless something below changes.

### Existing-test discipline

PR-I1 introduces no behavior change. The full pytest suite must run
on the `diffrax` branch with **identical results to `main`**: same
pass count, same xfail count, same residuals. If anything differs,
PR-I1 needs investigation before it merges.

## Verification

- `git tag --list | grep v0.1.0` shows the annotated tag with the
  documented message.
- On `diffrax` after PR-I1 merges: `pip install -e .` resolves
  `diffrax`; `python -c "from mam4_jax import solvers; print(solvers.SolverConfig())"`
  prints the dataclass.
- `python -m pytest tests/ -v` on `diffrax`: same line-for-line
  result as on `main` (73 passed, 6 xfailed at the time of writing,
  per `docs/PROGRESS.md`).
- ADR-014 reviewed and merged; ADR-013's status annotation reflects
  the partial supersession.
- `docs/HANDWRITTEN_SOLVER_LIMITATIONS.md` cross-checks against the
  `v0.1.0` tag message — they should describe the same baseline.

## What this PR does NOT do

- **No solver swap.** `_mam_soaexch_1subarea`, `_mam_gasaerexch_1subarea`,
  `_mam_coag_1subarea` are untouched. `solvers.py` exposes a
  `NotImplementedError`-raising skeleton.
- **No M5 xfail flip.** The 6 `nstep ≤ 30` cases stay `xfail` on
  `diffrax` after PR-I1; PR-D1 flips them.
- **No `main → diffrax` merge yet.** The first merge happens
  whenever there's a `main` change to bring across; not part of
  PR-I1's scope.
- **No CI changes** unless the CI config needs an explicit branch
  list for `diffrax`. Check during PR.

## Open questions

- **Diffrax minor-version pin.** Pick during PR (latest stable at
  the time of merge). If PR-D1 needs a newer release for a specific
  solver/feature, bump it in PR-D1 with a one-line rationale.
- **Should `solvers.py` live under `mam4_jax/` or
  `mam4_jax/processes/`?** Default proposal: top-level
  `mam4_jax/solvers.py`, peer of `kohler.py`, `coag.py`,
  `newnuc.py`, `constants.py` — it's infrastructure shared by
  multiple processes, not a process itself.
