# Deferred

Things explicitly **not** being done now, with a brief note on **why**. Anything listed here should either (a) move to `PLANS.md` when its time comes, or (b) be deliberately retired with a one-line note.

The point of this file is to keep the *decided to skip for now* knowledge out of people's heads.

---

## License selection

- **Status:** deferred.
- **Why:** No license file at the repo root yet (`README.md` notes this explicitly). The vendored Fortran reference retains its upstream license under `mam4-original-src-code/LICENSE`. Choosing a license for the JAX port (MIT? Apache-2.0? BSD-3-Clause?) is a decision that needs the owner + reflective-org alignment.
- **Resurface when:** before announcing the repo publicly or accepting external contributions.

## Stripping `__pycache__/*.pyc` from the vendored Fortran subtree

- **Status:** deferred.
- **Why:** The frozen snapshot includes two `.pyc` files at `mam4-original-src-code/postprocess/postprocess_scripts/__pycache__/`. They are upstream artifacts, not part of the Fortran reference proper. They were committed verbatim to preserve snapshot fidelity (ADR-001). Removing them is a one-line change but crosses the "don't modify the vendored snapshot" boundary; better to handle it as a deliberate, single-purpose PR than a drive-by.
- **Resurface when:** the first time someone refreshes the snapshot from upstream (handle as part of the refresh), or sooner if it bothers anyone.

## JAX package scaffolding

- **Status:** deferred until owner approves Milestone 1 in `PLANS.md`.
- **Why:** Three architectural decisions still need owner input (tracer representation, process signature style, configuration mechanism — see `ARCHITECTURE.md` "Open architectural questions"). Scaffolding without those decisions risks rework.
- **Resurface when:** ADR-007 through ADR-009 are landed in `KEY_DECISIONS.md`.

## CI / continuous integration

- **Status:** deferred.
- **Why:** No JAX code or tests exist yet; nothing meaningful to run in CI beyond markdown lint. Premature.
- **Resurface when:** Milestone 1 (JAX package scaffold) lands a first test.

## Differentiability through MAM4

- **Status:** deferred (explicitly).
- **Why:** A motivation for porting to JAX is potential autodiff through aerosol microphysics. However, several MAM4 routines use bisection, conditional branches on physical regimes, and other constructs that don't admit clean gradients out of the box. Promising differentiability up front would be a claim we cannot yet support.
- **Resurface when:** Milestone 6 (audit + optimization), as part of the differentiability audit subtask.

## Multi-column / multi-level support

- **Status:** deferred.
- **Why:** The Fortran reference is configured with `PCOLS=1`, `PVER=1` (single-column, single-level). The JAX port targets that configuration first. Generalizing to multiple columns/levels via `vmap` is straightforward but not the validation target.
- **Resurface when:** after Milestone 5 (convergence test reproduction) is green.

## GPU / TPU sharding

- **Status:** deferred.
- **Why:** Validation is the priority; CPU `float64` is the easiest path to a clean Fortran diff. Sharding decisions belong in Milestone 6 once correctness is established.
- **Resurface when:** Milestone 6.

---

*When adding a new deferred item: state what, why, and the condition that would bring it back.*
