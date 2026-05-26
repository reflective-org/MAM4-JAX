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

## ADR-012 — Fortran instrumentation lives outside the vendored tree, applied as a build-time overlay

- **Status:** Accepted (2026-05-18)
- **Context:** Validating each ported microphysics process against the Fortran reference requires capturing per-process inputs and outputs from a run. The reference is silent — `driver.F90` writes some scalars to text but not the full state arrays we need. We have two ways to get them: (a) edit the vendored `.F90` files in place, or (b) overlay instrumentation onto a transient build copy.
- **Decision:** Keep all instrumentation outside `mam4-original-src-code/`. The Fortran helper module (`scripts/patches/mam4_dump_state.F90`) and the unified-diff patch against `driver.F90` (`scripts/patches/driver_instrumentation.patch`) live under `scripts/patches/`. `scripts/build_reference.sh --instrumented` copies the helper into the transient `build/` directory and applies the patch there before invoking `make`, then overrides `OBJ9` so `mam4_dump_state.o` builds before `driver.o` and its `.mod` is in scope at link time. The committed `mam4-original-src-code/` tree is never modified.
- **Consequences:**
  - `git diff mam4-original-src-code/` is always empty, regardless of which build flavour was last run.
  - The patch must be maintained when the snapshot is refreshed (which would also bump the provenance commit in `README.md`). If anchor lines in `driver.F90` drift, the patch fails fast with `patch`'s standard hunk-rejection output.
  - One extra source file plus six `call dump_snapshot(...)` lines is enough overhead to capture full per-process I/O for all three top-level calls (`calcsize`, `wateruptake`, `amicphys`).
  - The instrumented binary writes intermediate `mam4_dump_<tag>.bin` records via Fortran stream I/O. `scripts/capture_reference.py --mode instrumented` parses those into committed `tests/reference/per_process/*.npz` archives. The `.bin` format is an implementation detail; `.npz` is the contract (`tests/reference/SCHEMA.md`).
- **Alternatives considered:**
  - Edit the vendored `.F90` files directly (rejected: violates ADR-001 and forces a "wash out instrumentation before refresh" workflow).
  - Use a separate driver that imports MAM4 modules and re-implements just enough of the time loop (rejected: duplicates 1600 lines of `driver.F90` orchestration logic that we'd then have to keep in sync).
  - Run the unmodified executable and post-process its existing text output (rejected: text output is incomplete and not numerically precise enough for `1e-6` validation).

---

## ADR-013 — Dual-branch strategy: `main` keeps handwritten solvers; `diffrax` branch ports them

- **Status:** Accepted (2026-05-22). Partially superseded by ADR-014 (2026-05-22): the "future `diffrax → main` merge is *not* anticipated" clause is reversed, and the cross-branch sync convention is replaced with merge-based sync. ADR-013's body is left unedited per the KEY_DECISIONS convention.
- **Context:** During M5 (12-point convergence sweep), the JAX port diverged sharply from Fortran at `nstep ≤ 30` (`dt ≥ 60s`). Diagnosis: Fortran's `mam_soaexch_1subarea` triggers adaptive substepping (`dtcur = alpha_astem/tmpa` at `modal_aero_amicphys.F90:3835-3843`); the JAX port intentionally assumes single-substep (deferred in M3.6 PR-E as PR-E2 per `docs/DEFERRED.md`). M7 (diffrax migration) was already on the roadmap as a future solver-quality improvement. The question: do we port handwritten adaptive substepping as PR-E2 first, then migrate to diffrax later, or skip PR-E2 and let the diffrax migration provide adaptive substepping for free?
- **Decision:** **Skip PR-E2 on `main`. Adaptive / dynamic substepping is solely the diffrax branch's responsibility.** A separate long-lived `diffrax` branch ports the handwritten ODE/analytical solvers (`_mam_gasaerexch_1subarea`'s H₂SO₄ analytical solver, `_mam_soaexch_1subarea`, possibly `_mam_coag_1subarea` if it has coupled-ODE structure) to diffrax equivalents. Diffrax's standard adaptive-controller (PI / I) handles step-size control natively — no handwritten substepping logic needed.
- **Branch invariants:**
  - **Structural parity.** Same module layout, function names, state-dict contract, and test fixtures on both branches. The only deltas are inside the affected solver bodies.
  - **Test parity.** The 6 currently-`xfail`ed `nstep ≤ 30` cases in `tests/test_sweep.py` should flip to expected-pass on the diffrax branch (with `rtol=1e-6` per ADR-003, possibly relaxed by 1 ULP if the solver choice shifts residuals).
  - **Cross-branch porting.** Non-solver changes (new docs, new fixtures, new processes) land in `main` first and then get cherry-picked / replayed into `diffrax`. Solver changes land in `diffrax` only. A future `diffrax → main` merge is *not* anticipated unless the project pivots to make diffrax the canonical implementation.
- **Consequences:**
  - `main` keeps the simpler, more transparent handwritten implementations — easier to read against Fortran 1:1.
  - `main` has a permanent gap on the `nstep ≤ 30` convergence-sweep cases. They stay `xfail` indefinitely with docstrings pointing at the diffrax branch as the resolution.
  - The two branches let us measure the diffrax change cleanly: same tests, same fixtures, only the solver differs. Performance / accuracy / autodiff trade-offs become directly comparable.
  - When the diffrax branch is ready to land, the decision to merge (or keep parallel) is itself a future ADR.
- **Alternatives considered:**
  - **Port PR-E2 (handwritten adaptive substepping) on `main`, then migrate to diffrax.** Rejected: duplicates work; tangles "match Fortran 1:1" with solver-quality improvements; the handwritten substepping would be deleted in M7 anyway.
  - **Skip M7 entirely; keep main as-is with the documented gap.** Rejected: the diffrax migration brings real benefits beyond adaptive substepping (autodiff cleanliness, standard error estimators, established library) — worth doing once we have a baseline.
  - **Apply the diffrax port as a series of PRs to `main` directly (no parallel branch).** Rejected: makes side-by-side comparison harder, forces both implementations to live in the same files, and loses the "structurally similar, only the solver differs" property the dual-branch arrangement provides.

---

## ADR-014 — Diffrax becomes canonical: eventual `diffrax → main` merge planned; sync via merge, not cherry-pick

- **Status:** Accepted (2026-05-22)
- **Context:** ADR-013 set up the dual-branch arrangement on the assumption that the `diffrax` branch might remain parallel indefinitely. In practice, the owner's intent (confirmed during M7 planning) is that once the diffrax port is validated end-to-end, it becomes the canonical MAM4-JAX implementation and merges back into `main`. ADR-013's "future `diffrax → main` merge is *not* anticipated" clause and its cherry-pick-based sync convention are inconsistent with that intent: a cherry-picked history makes the eventual merge-back noisier and the cross-branch comparison harder to maintain.
- **Decision:**
  1. **Eventual `diffrax → main` merge is planned**, not just possible. The diffrax branch is the canonical-to-be implementation. The decision *when* to merge — once all M7 sub-PRs (PR-I1, PR-D1, PR-D2, optionally PR-D3) land and the M5 `xfail`s flip — remains a future ADR; *that* it will eventually merge is settled.
  2. **Cross-branch sync uses periodic `main → diffrax` merges**, not cherry-picks, not rebases. Each baseline-sync is a merge commit on `diffrax` that brings the latest `main` (or, while `main` is gated by branch protection, the integration branch standing in for `main`) into `diffrax`. This preserves the full history so the eventual `diffrax → main` merge has a clean ancestry.
  3. **Solver changes still land on `diffrax` only.** ADR-013's "structural parity" invariant (same module layout, function names, state-dict contract, test fixtures) is preserved — the only deltas between branches are inside the solver bodies and any test-tolerance adjustments the merge-back will eventually unify.
  4. **The `v0.1.0` tag** on `main` (at the handwritten-solver baseline tip) anchors the pre-diffrax state. It is created out-of-band by the owner (not by automation) at a moment of their choosing — natural candidate: when this ADR merges. The tag's purpose is to preserve a checkout-able snapshot of the handwritten-solver implementation after the merge-back lands.
- **Consequences:**
  - `diffrax`'s history grows by accumulation of `main → diffrax` merges plus PR-D* solver-port commits; the eventual merge-back to `main` becomes a single (large) merge whose diff isolates the solver bodies.
  - `docs/HANDWRITTEN_SOLVER_LIMITATIONS.md` (introduced in PR-I1) documents what `v0.1.0` covers and doesn't, so users checking out the tag understand the gap.
  - The "permanent gap on `nstep ≤ 30`" wording in ADR-013 reads literally on `main` until the merge-back; after the merge-back, the gap is closed by diffrax and the `xfail`s are removed.
  - Branch protection on `main`: ADR-014 is agnostic. When `main` is gated, baseline syncs into `diffrax` flow from whichever branch carries the latest baseline (currently `dev`); the merge into `diffrax` still produces a merge commit with the right ancestry as long as the integration branch is a clean ancestor of `main`.
- **Alternatives considered:**
  - **Keep ADR-013's cherry-pick model.** Rejected: makes the eventual `diffrax → main` merge a manual reconciliation against a divergent history; defeats the "structurally similar, only the solver differs" promise.
  - **Rebase `diffrax` onto `main` periodically.** Rejected: rewrites `diffrax`'s tip, breaks anyone else tracking the branch, and discards the merge-commit ancestry that makes the future merge-back legible.
  - **Drop the merge-back intent; keep `diffrax` parallel forever.** Rejected: leaves two implementations of the same physics in the long term; doubles maintenance; the project would have to pick one as canonical eventually anyway.

---

## ADR-015 — Relaxed validation bar on the `diffrax` branch: <3% over 24h at dt≤5s, instead of ADR-003's 1e-6

- **Status:** Accepted (2026-05-25). Empirical revision: bar widened from initial 1% draft (2026-05-23) to 3% after the PR-D1 24h validation showed a dt-independent ~2.4% structural offset on `soag_gas`.
- **Context:** ADR-003 sets the project-wide JAX-vs-Fortran validation bar at `rtol=1e-6`. On `main`, this is achievable because the handwritten port and Fortran use the **same** semi-implicit scheme inside the same operator-splitting driver — the comparison is effectively implementation-identity, and the M5 sweep at `nstep ≥ 60` measures 1.97e-8 (bit-noise). On the `diffrax` branch (M7), `_mam_soaexch_1subarea` is replaced with a true-ODE adaptive integration (`Kvaerno5`). Diffrax produces a more physically correct soaexch result, but it differs from Fortran's semi-implicit by accumulated trajectory drift that turns out to be **larger and more structural than initially expected** — see PR-D1 empirical findings below.
- **Decision:**
  1. **The `diffrax` branch's JAX-vs-Fortran validation bar is `max rel-err < 3%` over a 24-hour box-model simulation at dt ≤ 5s.** Applies to all M7 sub-PRs (PR-D1, PR-D2, PR-D3).
  2. **Coarser dt (30s, 300s) is observational, not gated.** The validation suite reports rel-err at those dt but doesn't fail the build — the dt-dependence is documented as a known property of the operator-splitting truncation.
  3. **`main` retains ADR-003's `1e-6` bar.** The relaxation is `diffrax`-only. ADR-003 is unchanged.
  4. **The 24-hour fixture is the canonical validation surface.** New Fortran reference NetCDFs in `tests/reference/sweep_24h_no_pcarbon_aging/` capture the box model at dt ∈ {1s, 5s, 30s, 300s}. Tracked via git-lfs.
  5. **The relaxation reports per-field per-mode rel-err.** Per `project-mam4-per-mode-breakouts`, gas fields and per-mode aerosol fields are diagnosed independently; the 3% threshold applies to the max across all of them.
  6. **At the eventual `diffrax → main` merge-back, the bar question is reopened.** This ADR governs the `diffrax` branch in isolation.
- **Empirical context (PR-D1, 2026-05-25):** A 4-dt × 24-hour sweep showed:
  - `soag_gas` peak rel-err saturates at **~2.55% for dt ≤ 5s** (and the end-state rel-err at t=24h is ~2.42% across all tested dt — perfectly dt-independent). This is the structural offset between diffrax-true-ODE and Fortran-semi-implicit, not a bug; see `project-diffrax-structural-offset`.
  - Total active SOA mass drifts 0.35% heavier in JAX than Fortran by t=24h, also dt-independent. Mass conservation in H₂SO₄/SO4 and aerosol number is preserved to ~ε — the drift is SOA-specific.
  - `qgas_avg[0]` (SOA gas avg) was the leading suspect; tracing showed it is written by soaexch but **read by no downstream process**, so qgas_avg fixes cannot close the offset.
  - All other fields (num_aer, so4_aer, soa_aer per mode, h2so4_gas) pass under 1% at dt=5s. The 3% bar is set by `soag_gas` alone, with margin.
- **Why `3%`:** Empirical floor set by the soag_gas offset (~2.55% peak) plus ~0.5% margin. Tighter than 3% would force `soag_gas` to fail the validation; looser than 3% would erode the project's scientific-integrity value (4–5% on soag_gas would mean diffrax is materially less accurate than handwritten in some way the comparison missed).
- **Why `dt ≤ 5s`:** At coarser dt, operator-splitting truncation dominates (`soag_gas` peaks at 6.9% at dt=30s, 9.2% at dt=300s). Those errors are NOT diffrax-specific — both implementations suffer them — but ADR-003's machine-precision matching at coarse dt was an artifact of implementation-identity. With diffrax replacing one component, the operator-splitting truncation becomes visible. Improving it is M6 territory; ADR-015 doesn't gate on it.
- **Why `24 hours`:** Long enough to confirm rel-err saturation; matches a typical atmospheric column simulation window. The 30-min trajectories used during diagnosis showed plateau by t~300s but underestimated the long-time offset.
- **Consequences:**
  - The 12-point M5 convergence sweep's `rtol=1e-6` tests (`tests/test_sweep.py`) are rewritten for the diffrax branch — replaced by a 4-point 24h sweep parametrized over dt ∈ {1, 5, 30, 300}. dt=1 and dt=5 assert max rel-err < 3%; dt=30 and dt=300 record diagnostics without asserting. The 6 currently-`xfail`ed `nstep ≤ 30` cases are deleted — their failure mode (single-substep semi-implicit gap) doesn't apply on the diffrax branch.
  - The relaxation is **per-branch metadata**. Tooling that reads acceptance bars (CI, release scripts) must distinguish branch context.
  - PR-D2 (H₂SO₄ port) validates at the same 3% / 24h bar at dt ≤ 5s. PR-D2 may discover its own structural offsets; if so, the bar question reopens.
- **Alternatives considered:**
  - **Stick to 1%.** Rejected (2026-05-25 empirical refute): the `soag_gas` offset is ~2.4%, not 1%. A 1% bar fails on every PR-D1 case.
  - **Improve the driver's operator-splitting (Strang or higher-order) until diffrax matches at 1%.** Rejected: scope creep that conflates solver-port work with driver-architecture work; defers M7 by weeks; the operator-splitting work is M6's natural territory.
  - **Keep `rtol=1e-6` and replace diffrax with a custom semi-implicit-matching solver.** Rejected: defeats the purpose of moving to diffrax; the goal is physically-correct adaptive integration, not bit-exact Fortran reproduction.
  - **Field-specific bars (e.g., 3% on soag_gas, 1% elsewhere).** Rejected: simpler to apply a uniform bar and let per-field diagnostics speak for themselves in PR descriptions; field-specific bars create implicit branching in the test suite.
  - **Test at the END of 24h only (rel-err at t=24h, not peak).** Rejected: peak rel-err captures the transient excursion (a physically meaningful diagnostic); end-state-only would hide the transient and would be a weaker validation.

---

*Add new ADRs below this line. Number sequentially; never reuse numbers; never edit an Accepted ADR.*
