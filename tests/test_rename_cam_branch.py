"""CAM's rename algorithm, alongside E3SM's.

CAM's default path (`modal_aero_rename_no_acc_crs_sub`) and E3SM's production
path (`mam_rename_1subarea` with `rename_method_optaa = 40`) are a real
algorithmic difference — so a CAM driver cannot reuse the existing port.

The useful discovery is that E3SM's code *contains* CAM's algorithm as its
`optaa /= 40` branch, and CAM's path is line-for-line identical to E3SM's legacy
rename (27 differing lines out of 243, none touching the arithmetic). So this is
one selector, not a second implementation. Three decisions differ
(`mam-box-fortran/docs/reference/discrepancies/rename.md` §4):

1. whether the growth increment is gated up front,
2. what triggers the old-diameter clamp,
3. whether the old *volume* is rescaled with the clamped diameter.

WHAT IS AND IS NOT VALIDATED HERE. These tests establish that the CAM branch is
reachable, distinct, conservative, and shows CAM's documented gate behaviour.
They do **not** establish numeric agreement with the Fortran — that needs the
unit and index mapping between CAM's `q`/`dqdt` convention and this module's
amicphys-local view, and a reference set exists for it in `mam-box-fortran`
(`tools/capture_rename`, `outputs/rename_ref/`). That comparison is the next
step and is deliberately not claimed yet.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import mam4_jax  # noqa: F401
from mam4_jax.coupling.amicphys import _mam_rename_1subarea

REF = Path(__file__).resolve().parent / "reference" / "per_process"
AITKEN = 1          # mode order in the amicphys-local view: accum, aitken, coarse, pcarbon
OVERSIZE = 0.01     # scale on aitken number; smaller = fewer, larger particles


@pytest.fixture(scope="module")
def local_view():
    b = np.load(REF / "rename_before.npz", allow_pickle=False)
    return {k: np.asarray(b[k][0]) for k in
            ("qnum_cur", "qaer_cur", "qaer_delsub_grow4rnam", "qwtr_cur",
             "fac_m2v_aer")}


def _call(v, *, method, num_scale=1.0, growth_scale=1.0):
    qn = v["qnum_cur"].copy()
    qn[AITKEN] *= num_scale
    return _mam_rename_1subarea(
        jnp.asarray(qn),
        jnp.asarray(v["qaer_cur"]),
        jnp.asarray(v["qaer_delsub_grow4rnam"] * growth_scale),
        jnp.asarray(v["qwtr_cur"]),
        jnp.asarray(v["fac_m2v_aer"]),
        method=method,
    )


def test_invalid_method_is_rejected(local_view) -> None:
    with pytest.raises(ValueError, match="must be 'cam' or 'e3sm'"):
        _call(local_view, method="optaa40")


def test_default_is_e3sm_so_existing_behaviour_is_unchanged(local_view) -> None:
    """The default must stay E3SM: this is a production port and the CAM branch
    is new. test_rename.py's Fortran comparison also depends on it."""
    a = _call(local_view, method="e3sm")
    b = _mam_rename_1subarea(
        jnp.asarray(local_view["qnum_cur"]),
        jnp.asarray(local_view["qaer_cur"]),
        jnp.asarray(local_view["qaer_delsub_grow4rnam"]),
        jnp.asarray(local_view["qwtr_cur"]),
        jnp.asarray(local_view["fac_m2v_aer"]),
    )
    for x, y in zip(a, b):
        assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


def test_the_reference_fixture_alone_cannot_tell_the_branches_apart(local_view) -> None:
    """Worth pinning: at the captured state the transfer barely engages and the
    two algorithms agree exactly. A test suite built only on this fixture would
    conclude the CAM branch was a no-op."""
    cam = _call(local_view, method="cam")
    e3sm = _call(local_view, method="e3sm")
    assert np.allclose(np.asarray(cam[0]), np.asarray(e3sm[0]), rtol=0, atol=0)


def test_the_branches_diverge_once_the_mode_is_oversized(local_view) -> None:
    """With aitken oversized the old diameter clears `dp_belowcut`, which is
    where the clamp and volume-rescale differences bite."""
    cam = _call(local_view, method="cam", num_scale=OVERSIZE)
    e3sm = _call(local_view, method="e3sm", num_scale=OVERSIZE)
    dn = float(np.max(np.abs(np.asarray(cam[0]) - np.asarray(e3sm[0]))))
    assert dn > 0.0, "CAM and E3SM rename must differ in the oversized regime"


def test_cam_gates_on_growth_and_e3sm_does_not(local_view) -> None:
    """The single most discriminating behaviour.

    CAM gates up front on `dryvol_t_del <= 1e-6 * dryvol_t_oldbnd`, so with no
    growth it transfers nothing however oversized the mode is. E3SM skips that
    gate (`optaa == 40`) and applies a different one after rescaling. The
    Fortran capture shows the same thing: growth 0 gives a zero transfer, and
    that was the first thing the capture tool got wrong.
    """
    qn_ait = local_view["qnum_cur"][AITKEN] * OVERSIZE

    cam = _call(local_view, method="cam", num_scale=OVERSIZE, growth_scale=0.0)
    moved_cam = abs(float(np.asarray(cam[0])[AITKEN]) - qn_ait)
    assert moved_cam == 0.0, f"CAM must not transfer without growth; moved {moved_cam:.3e}"

    e3sm = _call(local_view, method="e3sm", num_scale=OVERSIZE, growth_scale=0.0)
    moved_e3sm = abs(float(np.asarray(e3sm[0])[AITKEN]) - qn_ait)
    assert moved_e3sm > 0.0, (
        "E3SM skips the up-front growth gate, so it should still transfer; if "
        "this fails the branches are no longer distinguished by that gate"
    )


@pytest.mark.parametrize("method", ["cam", "e3sm"])
def test_number_and_mass_are_conserved(method: str, local_view) -> None:
    """Rename moves material between modes; totals must not change.

    Note the baseline is `qaer_cur` ALONE, not `qaer_cur + qaer_delsub_grow4rnam`.
    The growth is already folded into `qaer_cur`; the delta is informational,
    used only to form `dryvol_t_del`. Checked against the Fortran reference
    itself, which conserves against `qaer_cur` to 0.0 and against
    `qaer_cur + delta` to 2.4e-3 — so the wrong baseline reads as a 0.24 % leak.
    """
    qn = local_view["qnum_cur"].copy()
    qn[AITKEN] *= OVERSIZE
    out = _call(local_view, method=method, num_scale=OVERSIZE)

    n0, n1 = qn.sum(), float(np.asarray(out[0]).sum())
    assert abs(n1 - n0) / n0 < 1e-12, f"number not conserved: {n0:.6e} -> {n1:.6e}"

    a0 = local_view["qaer_cur"].sum()
    a1 = float(np.asarray(out[1]).sum())
    assert abs(a1 - a0) / a0 < 1e-12, f"aerosol not conserved: {a0:.6e} -> {a1:.6e}"


def test_cam_transfer_grows_then_saturates_with_growth(local_view) -> None:
    """Signature the Fortran capture showed: the transferred number saturates
    (the tail beyond the boundary is finite) while the driver keeps increasing.
    A port that scaled without bound would be wrong in a way conservation and
    positivity checks cannot see.
    """
    qn_ait = local_view["qnum_cur"][AITKEN] * OVERSIZE
    moved = []
    for g in (0.0, 0.5, 2.0, 20.0, 200.0):
        out = _call(local_view, method="cam", num_scale=OVERSIZE, growth_scale=g)
        moved.append(abs(float(np.asarray(out[0])[AITKEN]) - qn_ait))

    assert moved[0] == 0.0                       # gated at zero growth
    assert all(b >= a for a, b in zip(moved, moved[1:])), f"not monotone: {moved}"
    assert moved[-1] <= qn_ait, "cannot transfer more particles than exist"
    # Saturation: the last doubling of growth must move far less than the first.
    assert (moved[-1] - moved[-2]) < (moved[2] - moved[1]), f"no saturation: {moved}"
