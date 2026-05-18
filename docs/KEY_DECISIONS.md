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

- **Status:** Superseded by ADR-011 (2026-05-18)
- **Context:** Rule #2 in `CLAUDE.md` calls for PR-per-task. At pre-launch with one contributor, that overhead delivers little.
- **Decision:** Allow direct commits to `main` for scaffolding/documentation work only, while the project has no external readers. Process and feature changes still go through PRs. Revisit (and supersede this ADR) when contributors join or the first JAX code lands.
- **Consequences:**
  - Faster initial setup.
  - Lower historical traceability for the scaffolding phase; mitigated by detailed `PROGRESS.md` entries.
- **Alternatives considered:** Strict PR-per-commit (rejected: friction not justified yet).

## ADR-007 — Plans are archived under `docs/plans/NNN-<slug>.md`

- **Status:** Accepted (2026-05-18)
- **Context:** Planning sessions produce a single canonical plan file at a transient location (`~/.claude/plans/...`). When the plan is approved, the project loses access to that record unless it's checked in. We need the planning history to live alongside the project docs so contributors can read what was decided and why.
- **Decision:** When a plan is approved, copy it verbatim to `docs/plans/NNN-<slug>.md` (zero-padded sequential numbering, kebab-case slug derived from the plan title). The copy is part of the same PR/commit that begins executing the plan. If post-approval edits to the plan are necessary (e.g., ADR numbering shifts), add an editorial note at the top of the archived plan rather than rewriting the body.
- **Consequences:**
  - The `docs/plans/` directory is an append-only log of approved plans.
  - Plans referenced from `docs/PROGRESS.md` and `docs/PLANS.md` link to their `docs/plans/NNN-...` archive.
  - Plan numbering is independent of ADR numbering.
- **Alternatives considered:** keep plans only in the transient location (rejected: no project-visible history); rewrite plans after approval to keep them "current" (rejected: dilutes the historical record).

## ADR-008 — Tracer representation: flat `pcnst` array with named accessors

- **Status:** Accepted (2026-05-18). Approved as ADR-007 in `docs/plans/001-scaffold-and-reference-capture.md`; renumbered here to 008 because ADR-007 was claimed by the docs/plans/ convention before this PR landed.
- **Context:** The Fortran reference passes a flat `q(:,:,pcnst)` tracer array everywhere, with `modal_aero_data` integer index tables (`numptr_amode`, `lmassptr_amode`, …) mapping (mode, species_slot) → `pcnst` index. A JAX port could either mirror this flat layout or restructure as a per-(mode, species) pytree.
- **Decision:** Mirror the Fortran flat layout. Primary state is a JAX array of shape `(pcols, pver, pcnst)`. Index tables and accessor helpers (`get_number`, `get_mass`) live in `mam4_jax/data.py` and are the only place that resolves a (mode, species_slot) to a `pcnst` index.
- **Consequences:**
  - Byte-for-byte diffing against the Fortran reference is straightforward; no per-axis translation is needed.
  - The index bookkeeping is concentrated in one module, surfacing what would otherwise be the single largest source of porting bugs.
  - JAX code looks less "JAX-native" than a pytree-of-arrays approach would.
- **Alternatives considered:** Per-(mode, species) pytree (rejected: obscures Fortran correspondence during validation, harder to diff at the 1e-6 tolerance set in ADR-003).

## ADR-009 — Process signature convention: pure-functional

- **Status:** Accepted (2026-05-18). Approved as ADR-008 in `docs/plans/001`; renumbered to 009 here.
- **Context:** Fortran microphysics subroutines (e.g., `modal_aero_wateruptake_dr` at `modal_aero_wateruptake.F90:130-150`) mutate state via pointer arguments and side effects. That convention is the opposite of how JAX wants to see code: `jit`, `vmap`, `scan`, and the autodiff system require pure functions.
- **Decision:** Every microphysics function in `mam4_jax/processes/` has the signature `process_fn(state, params, config) -> new_state`. No in-place mutation. No pointer-output args. Configuration enters as a dataclass (ADR-010), not module-level globals.
- **Consequences:**
  - JAX code is structurally different from the Fortran. That is expected and tolerable; correctness via numerical diff (ADR-003) is the constraint, not structural fidelity.
  - The Phase B optimization pass (ADR-004: `jit`/`vmap`/`scan`) becomes mechanical because the inputs are already pure.
  - Per-process testing is simplified: each function is fully determined by its inputs.
- **Alternatives considered:** Mutable-state convention via dataclass-with-setters or `numpy` in-place ops (rejected: incompatible with `jit`/autodiff; would force a rewrite in Phase B).

## ADR-010 — Configuration: Python `dataclass`es with optional YAML loader

- **Status:** Accepted (2026-05-18). Approved as ADR-009 in `docs/plans/001`; renumbered to 010 here.
- **Context:** The Fortran reference takes input through four namelist groups (`&time_input`, `&cntl_input`, `&met_input`, `&chem_input`) of scalar parameters. The JAX port needs an equivalent input surface for reproducibility tests and for sweeping over inputs.
- **Decision:** Four frozen `@dataclass` types (`TimeConfig`, `ControlConfig`, `MetConfig`, `ChemConfig`) mirror the namelist groups one-to-one, with field names taken verbatim from the namelist symbols. A `RunConfig` composite groups all four. `load_yaml(path)` returns a `RunConfig` from a YAML file; defaults reflect `run_test.csh`'s canonical inputs.
- **Consequences:**
  - Configuration is type-safe and immutable.
  - YAML config files diff cleanly against the namelist blocks in `driver.F90`, supporting reproducible experiment archives.
  - Field name fidelity to Fortran symbols (e.g., `mdo_gasaerexch`, `mfso41`) preserves provenance at the cost of un-Pythonic field names. Accepted.
- **Alternatives considered:** Free-form dict (rejected: no type checking, lossy), TOML (rejected: less common in the scientific-Python ecosystem and we already depend on YAML elsewhere), `pyrallis`/`hydra` (rejected: heavyweight for a 4-namelist input surface).

## ADR-011 — All changes via pull request; ADR-006 superseded

- **Status:** Accepted (2026-05-18). **Supersedes ADR-006.**
- **Context:** ADR-006 carved out "direct commits to `main`" for scaffolding/docs work during pre-launch. In practice this turned out to be ambiguous (where exactly does scaffolding end and feature work begin?) and the local auto-classifier correctly refused direct pushes against rule #2. Rather than relax the rule, tighten it: rule #2 applies uniformly.
- **Decision:** All changes to `main` go through a PR. No carve-out for "scaffolding" or "docs only." For solo work the PR overhead is small (`gh pr create`, self-review, merge); the win is uniform process and a clean reviewable history.
- **Consequences:**
  - Local `main` should never carry unpushed commits. If a one-off direct commit is unavoidable (e.g., a `.gitignore` typo), it still gets a branch and a PR.
  - Pre-existing `main` history (commits `22f212d`, `a82e42d` from initial setup) stays as-is — this ADR is forward-looking, not retroactive.
- **Alternatives considered:** Keep ADR-006 and clarify the carve-out (rejected: ambiguity itself is the problem); add a manual override flag (rejected: no value).

---

*Add new ADRs below this line. Number sequentially; never reuse numbers; never edit an Accepted ADR.*
