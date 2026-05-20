"""End-to-end validation of the JAX port of ``mam_rename_1subarea``.

For each captured timestep of the full-physics Fortran reference
(``tests/reference/per_process/rename_{before,after}.npz``), call the
JAX port on the "before" snapshot and assert the result matches the
"after" snapshot at ADR-003's 1e-6 relative tolerance.

The reference was captured via the ADR-012 instrumentation overlay
extended with ``scripts/patches/rename_hook.patch`` — see
``docs/plans/002-rename-port.md`` for why we validate against the
full-physics fixture rather than from the JAX orchestration shell.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from mam4_jax import data
from mam4_jax.processes.amicphys import _mam_rename_1subarea

REF_DIR = Path(__file__).resolve().parent / "reference" / "per_process"


@pytest.fixture(scope="module")
def rename_reference() -> tuple[dict, dict]:
    before = np.load(REF_DIR / "rename_before.npz", allow_pickle=False)
    after  = np.load(REF_DIR / "rename_after.npz",  allow_pickle=False)
    return dict(before), dict(after)


def test_rename_matches_fortran_full_physics(rename_reference) -> None:
    """Per-step diff of the JAX port against the Fortran reference."""
    before, after = rename_reference
    nstep = before["istep"].shape[0]
    assert nstep == after["istep"].shape[0] == 60

    rel_qnum_max = 0.0
    rel_qaer_max = 0.0

    for t in range(nstep):
        qnum_in  = jnp.asarray(before["qnum_cur"][t])
        qaer_in  = jnp.asarray(before["qaer_cur"][t])
        qdel_in  = jnp.asarray(before["qaer_delsub_grow4rnam"][t])
        qwtr_in  = jnp.asarray(before["qwtr_cur"][t])
        fac_in   = jnp.asarray(before["fac_m2v_aer"][t])

        qnum_out, qaer_out, qwtr_out = _mam_rename_1subarea(
            qnum_in, qaer_in, qdel_in, qwtr_in, fac_in,
        )

        qnum_ref = after["qnum_cur"][t]
        qaer_ref = after["qaer_cur"][t]
        qwtr_ref = after["qwtr_cur"][t]

        # qnum_cur: per-mode, can be zero (unused mode slots) → absorb the
        # zero-denominator entries with a small absolute floor.
        np.testing.assert_allclose(
            np.asarray(qnum_out), qnum_ref,
            rtol=1e-6, atol=1e-25,
            err_msg=f"qnum_cur mismatch at step {t + 1}",
        )
        np.testing.assert_allclose(
            np.asarray(qaer_out), qaer_ref,
            rtol=1e-6, atol=1e-25,
            err_msg=f"qaer_cur mismatch at step {t + 1}",
        )
        # qwtr_cur is intent(inout) but unused by rename — JAX preserves
        # it; the Fortran reference's "after" should equal "before".
        np.testing.assert_array_equal(np.asarray(qwtr_out), qwtr_ref)

        # Track tightest matches for the diagnostic print below.
        denom_qnum = np.maximum(np.abs(qnum_ref), 1e-25)
        denom_qaer = np.maximum(np.abs(qaer_ref), 1e-25)
        rel_qnum_max = max(rel_qnum_max,
                           float(np.max(np.abs(np.asarray(qnum_out) - qnum_ref) / denom_qnum)))
        rel_qaer_max = max(rel_qaer_max,
                           float(np.max(np.abs(np.asarray(qaer_out) - qaer_ref) / denom_qaer)))

    # Print the tightest relative-error envelope so the test log captures
    # how far below 1e-6 the port actually sits.
    print(f"\nrel-err envelope across 60 steps:"
          f"  qnum max = {rel_qnum_max:.3e},  qaer max = {rel_qaer_max:.3e}")


def test_rename_conserves_number_and_mass(rename_reference) -> None:
    """Rename only redistributes — total number and per-species mass are conserved.

    Asserted across all 60 captured steps. The Aitken→accum transfer
    subtracts ``dnum`` from one mode and adds the same value to the
    other; same for each species' mass. So summing over modes must yield
    the same total in "before" and "after".

    Stronger than a pure structural test: it would catch any sign error
    in the ``.at[].add()`` plumbing without relying on Fortran reference
    data.
    """
    before, after = rename_reference
    nstep = before["istep"].shape[0]

    for t in range(nstep):
        qnum_in  = jnp.asarray(before["qnum_cur"][t])
        qaer_in  = jnp.asarray(before["qaer_cur"][t])
        qdel_in  = jnp.asarray(before["qaer_delsub_grow4rnam"][t])
        qwtr_in  = jnp.asarray(before["qwtr_cur"][t])
        fac_in   = jnp.asarray(before["fac_m2v_aer"][t])

        qnum_out, qaer_out, _ = _mam_rename_1subarea(
            qnum_in, qaer_in, qdel_in, qwtr_in, fac_in,
        )

        # Total number (summed over modes) is conserved.
        np.testing.assert_allclose(
            float(jnp.sum(qnum_out)), float(jnp.sum(qnum_in)),
            rtol=1e-14, atol=0.0,
            err_msg=f"qnum total changed at step {t + 1}",
        )
        # Total mass per species (summed over modes) is conserved.
        np.testing.assert_allclose(
            np.asarray(jnp.sum(qaer_out, axis=1)),
            np.asarray(jnp.sum(qaer_in,  axis=1)),
            rtol=1e-14, atol=1e-30,
            err_msg=f"qaer per-species total changed at step {t + 1}",
        )
