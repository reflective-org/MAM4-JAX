# Plan 018 — M6 PR-J1: `jax.jit` boundaries on the diffrax branch

> **Status:** proposed 2026-05-26. Awaiting owner approval before
> implementation.

---

## Context

ADR-016 (2026-05-26) fixed M6 as the next milestone on `diffrax`,
to run before the `diffrax → main` merge-back. M6 PR-J1 is the
first M6 sub-PR.

The diffrax port is currently uncompiled at the call-site level.
`mam4_jax/solvers.py::solve_ivp` calls `diffrax.diffeqsolve`,
which JITs *internally* (via diffrax's adaptive controller and
solver kernels), but the wrapping code — `solve_ivp` itself, the
two RHS functions (`_h2so4_rhs`, `_soaexch_rhs`), the orchestrator
(`_mam_amicphys_1subarea_clear`), and the driver step (`run_step`)
— runs as eager Python.

PR-D2 measured the practical impact: 24 h at dt=1 s takes
~2270 s (38 min) of JAX wall time, dominated by ~86 400 outer
driver iterations of un-JIT-ted Python orchestration. PR-D1's
note "uncompiled diffrax is ~50× slower than handwritten" sets
the expected payoff: a JIT-compiled run should drop dt=1 s 24 h
from minutes to seconds.

This PR introduces `@jax.jit` decorators at the right boundaries
and validates the result.

## Where to put `@jax.jit`

Candidates, listed bottom-up:

- **`solve_ivp`** — already mostly JIT-clean (diffrax handles the
  internal). Wrapping it with `@jax.jit` is mostly redundant but
  cheap; useful if other call sites want a stable trace boundary
  to JIT around. Decide empirically.
- **`_h2so4_rhs`, `_soaexch_rhs`** — pure jnp functions, already
  JIT-clean. No `@jax.jit` needed at module level; they get
  traced when `solve_ivp` is called.
- **Per-process `*_1subarea` functions** (e.g. `_mam_soaexch_1subarea`,
  `_mam_gasaerexch_1subarea`, `_mam_newnuc_1subarea`,
  `_mam_coag_1subarea`, `_mam_rename_1subarea`) — these are pure
  functional. JIT-compiling each at module level might help if
  they're called repeatedly (they are — once per outer driver
  step), but the wider orchestrator is the better boundary.
- **`_mam_amicphys_1subarea_clear`** (the orchestration shell) —
  natural unit of work per (col, level) per substep. Pure-functional
  (returns a new state dict). JIT-compiling this is the right
  primary boundary.
- **`run_step`** — drives one full operator-splitting step (calcsize
  → wateruptake → amicphys → ...). Wrapping with `@jax.jit` JITs
  the whole step. Probably **the load-bearing boundary** because
  it's what gets called in the time loop.
- **`run_timesteps`** — the time loop itself. Currently a Python
  `for` loop. JIT-compiling the loop body (i.e., `run_step`)
  is PR-J1's scope; replacing the loop with `jax.lax.scan`
  is **PR-J2**, separately.

**Default proposal for PR-J1:** `@jax.jit` on `run_step` only, plus
any minor adjustments needed for the trace to succeed (e.g.,
static_argnums for non-traced kwargs). Other boundaries get JITed
indirectly when `run_step` traces them.

## Risks

- **State-dict tracing.** The state dict is the natural pytree;
  JAX traces through it cleanly if the dict structure is stable.
  Verify the trace succeeds without explicit `register_pytree_node`
  hacks. If the state-dict shape varies per step (it shouldn't —
  `_repack_amicphys_view_to_state` returns the same keys), this
  is straightforward.
- **Compile-time spike on first call.** Tracing `run_step` once
  through all the diffrax solvers, the unpack/repack, all sub-
  processes, etc. will likely take 10–30 s on first call. That's
  fine as long as it's amortised over a real run. Worth benchmarking
  and quoting in the PR description.
- **JIT-incompatible Python.** If any sub-process has a Python-
  level `if`/`for` on a traced value, the trace fails. PR-J4's
  cond/where audit was scheduled separately for this reason; if
  PR-J1's trace surfaces a problem, do a minimal in-PR fix and
  defer broader auditing to PR-J4.

## Validation

The 4-dt × 24 h sweep is the canonical validation surface (per
PR-D2). After PR-J1:

| dt (s) | overall max (should match PR-D2) | wall time (should drop) |
| --- | --- | --- |
| 1 | 2.55 % ± solver noise (1 ULP) | ~80 min → expect a few seconds |
| 5 | 2.55 % | ~8 min → expect <1 s |
| 30 | 6.91 % | ~75 s → expect <1 s |
| 300 | 9.21 % | ~11 s → expect <1 s |

**Hard acceptance criteria:**

1. `tests/test_sweep.py[1]` and `[5]` continue to pass at the
   3 % bar (ADR-015). Per-mode rel-err breakouts should match
   PR-D2's to within solver-noise levels (any change >1 ULP gets
   called out in the PR description).
2. `tests/test_sweep.py[30]` and `[300]` continue to print
   diagnostic rel-err that matches PR-D2's to within 1 ULP.
3. All other tests (`tests/test_*.py`) continue to pass at their
   existing bars (1e-6 on most).
4. Wall time on dt=1 s 24 h drops by at least **10×** (target:
   `> 100×`).
5. First-call compile time is <30 s.

## What this PR does NOT do

- **No `jax.lax.scan`** — that's PR-J2.
- **No `jax.vmap`** — that's PR-J3.
- **No autodiff audit** — that's PR-J5.
- **No new physics, no new tests, no new fixtures.** Pure
  optimization PR. Diff stays small.

## Open questions

- **JIT cache strategy.** `jax.jit` caches by abstract value
  (shape + dtype). The state dict has many entries; if any of
  them have shape variation across calls, the cache misses each
  time. Sanity-check that all state-dict entries are
  shape-stable. Plan: trace `run_step` twice in a row with
  identical input shapes; second call should be sub-millisecond.
- **`static_argnums` on `run_step`.** Things like `mdo_*` flags,
  `params`/`config` — if any of those flow into the trace, they
  may need `static_argnums` or `static_argnames`. Decide
  empirically; default is no static args.
- **`scripts/diffrax_24h_validation.py` rewrite?** The cached
  `.npz` outputs are the inputs to the plot scripts; if PR-J1
  changes the JAX numerical output by even 1 ULP, the cached
  `.npz` files need regenerating. Default: regenerate the cache
  + plots as part of the PR.
