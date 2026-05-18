# Key Decisions

Architecture Decision Records (ADRs). Each ADR captures the *why* behind a load-bearing choice; the corresponding *rule* lives in `CLAUDE.md`. Append new ADRs at the bottom — never edit accepted ADRs in place; instead, add a new ADR that supersedes the old one.

Status values: **Accepted**, **Proposed**, **Superseded by ADR-NNN**.

---

## ADR-001 — Vendor MAM4 Fortran reference as a frozen snapshot

- **Status:** Accepted (2026-05-18)
- **Context:** The new repo needs the Fortran reference available to (a) run validation, (b) let new contributors clone-and-go. The reference originates upstream at `kaizhangpnl/MAM_box_model` and was already cloned into a `reflective-org/MAM4_box_model` fork.
- **Decision:** Copy the source files into `mam4-original-src-code/` without a nested `.git/` directory or submodule. Record the upstream URL, commit SHA, and snapshot date in `README.md`.
- **Consequences:**
  - `git clone` of MAM4-JAX gives everything needed to validate without submodule incantations.
  - Pulling future upstream changes is a manual replace + provenance-table update, not an automatic merge.
  - Files under `mam4-original-src-code/` are treated as read-only; any modification must be deliberate and noted.
- **Alternatives considered:** git submodule (rejected: clone friction), git subtree (rejected: history-merge complexity for a frozen reference), don't commit it / `gitignore` (rejected: breaks clone-and-go).

## ADR-002 — Default numerical precision is `float64`

- **Status:** Accepted (2026-05-18)
- **Context:** MAM4 is stiff; ratios of species concentrations span many orders of magnitude; nucleation kinetics and water-uptake equilibrium are sensitive to round-off. JAX defaults to `float32` unless explicitly enabled.
- **Decision:** The JAX package will call `jax.config.update("jax_enable_x64", True)` at import time and use `float64` for all aerosol/gas state. Lower precision is allowed only at clearly bounded performance-critical leaves with documented justification.
- **Consequences:**
  - Some GPU/TPU configurations are slower in `float64`; this is acceptable for a validation-first port.
  - Mixing `float32` and `float64` in JAX silently upcasts; tests must assert dtypes explicitly.
- **Alternatives considered:** `float32` default with selective promotion (rejected: too easy to lose precision silently in a scientific port).

## ADR-003 — Validation tolerance is `1e-6` relative error

- **Status:** Accepted (2026-05-18)
- **Context:** "Match Fortran exactly" is unachievable across compilers, transcendentals, and reduction orders. A bound is needed.
- **Decision:** Element-wise maximum relative error between the JAX port and the Fortran reference must be below `1e-6` for each ported subroutine, with appropriate handling of small-magnitude values (absolute-tolerance floor TBD per process). Tolerances are tightened where physically meaningful and may be relaxed only with explicit owner approval, recorded as a new ADR.
- **Consequences:**
  - Most pure-arithmetic ports should hit this comfortably; iterative/bisection routines may need careful initial-condition matching.
  - Each ported PR must include the measured rel-err alongside the assertion.
- **Alternatives considered:** Bit-exact (rejected: not portable), `1e-3` (rejected: too loose for stiff microphysics), absolute-only tolerance (rejected: mode-number concentrations span many orders of magnitude).

## ADR-004 — Two-phase port: scaffold first, optimize later

- **Status:** Accepted (2026-05-18)
- **Context:** Aggressive use of `jit`, `vmap`, `scan`, `cond` during initial porting tends to obscure bugs and slow validation. JAX-idiomatic optimization is best applied to code that is already known to be correct.
- **Decision:** Phase A is a straightforward Python-loop, eager-execution port of each process, validated against the Fortran reference. Phase B is a sweep applying `jit`/`vmap`/`scan`/sharding, with the rel-err threshold reasserted after each change.
- **Consequences:**
  - Phase A code looks unidiomatic for JAX. That's expected and temporary.
  - Phase B is its own milestone (`PLANS.md` Milestone 6) with separate PRs per optimization.
- **Alternatives considered:** Optimize-as-you-port (rejected: conflates correctness and performance bugs).

## ADR-005 — Documentation lives under `docs/`, extracted from `CLAUDE.md`

- **Status:** Accepted (2026-05-18)
- **Context:** `CLAUDE.md` had grown to include rules, architecture, and embedded design decisions. Mixing agent-facing rules with reference material made both harder to navigate.
- **Decision:** Move architecture details into `docs/ARCHITECTURE.md` and design rationales into `docs/KEY_DECISIONS.md`. `CLAUDE.md` keeps binding rules, behavioral guardrails, validation workflow, and short pointers into the docs. Add `docs/PROGRESS.md`, `docs/PLANS.md`, `docs/DEFERRED.md`, `docs/FEATURES.md` for ongoing project state.
- **Consequences:**
  - Single source of truth per topic; no duplication.
  - `CLAUDE.md` stays compact and agent-readable.
  - Contributors must update the relevant doc in the same PR that introduces the change it describes.
- **Alternatives considered:** Keep `CLAUDE.md` as the sole canonical doc (rejected: doesn't scale), duplicate into both (rejected: drift).

## ADR-006 — Direct commits to `main` during pre-launch scaffolding

- **Status:** Accepted (2026-05-18)
- **Context:** Rule #2 in `CLAUDE.md` calls for PR-per-task. At pre-launch with one contributor, that overhead delivers little.
- **Decision:** Allow direct commits to `main` for scaffolding/documentation work only, while the project has no external readers. Process and feature changes still go through PRs. Revisit (and supersede this ADR) when contributors join or the first JAX code lands.
- **Consequences:**
  - Faster initial setup.
  - Lower historical traceability for the scaffolding phase; mitigated by detailed `PROGRESS.md` entries.
- **Alternatives considered:** Strict PR-per-commit (rejected: friction not justified yet).

---

*Add new ADRs below this line. Number sequentially; never reuse numbers; never edit an Accepted ADR.*
