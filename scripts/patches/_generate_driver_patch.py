"""Generator for scripts/patches/driver_instrumentation.patch.

This script is NOT used at build time — the canonical instrumentation patch is
the static unified diff at ``scripts/patches/driver_instrumentation.patch``,
which ``scripts/build_reference.sh --instrumented`` applies with ``patch -p1``.

The static patch was produced by:
  1. Copying ``mam4-original-src-code/test_drivers/driver.F90`` to a temp file.
  2. Running this script against the copy.
  3. ``diff -u`` between the original and the modified copy.
  4. Rewriting the diff headers to ``a/driver.F90`` / ``b/driver.F90``.

Use this script to regenerate the patch when the upstream Fortran snapshot is
refreshed and line numbers shift. Anchors are text-based, so as long as the
anchor lines still appear verbatim in the new source the regeneration
succeeds; otherwise update the anchors below and re-run.

Inserts a ``use mam4_dump_state, only: dump_snapshot`` clause inside the
``cambox_do_run`` subroutine plus six ``call dump_snapshot(...)`` invocations
around the three top-level microphysics calls. Idempotent: bails if the hooks
already appear in the file.
"""
import sys
from pathlib import Path

DUMP_FIELDS = (
    "istep, ncol, pver, pcnst, ntot_amode, &\n"
    "         q, qqcw, dgncur_a, dgncur_awet, qaerwat, wetdens"
)


def dump_call(tag: str) -> str:
    return (
        f"      call dump_snapshot('{tag}', {DUMP_FIELDS})\n"
        "\n"
    )


USE_LINE = "      use mam4_dump_state, only: dump_snapshot\n"


# Anchors: (description, exact source line, where_to_insert, payload).
# where_to_insert ∈ {"before", "after"}.
ANCHORS = [
    # Add USE clause inside cambox_do_run subroutine. The anchor is the
    # cambox_do_run-local `use modal_aero_wateruptake, only:
    # modal_aero_wateruptake_dr` line — distinct from the same module's
    # use in cambox_init_basics (which imports different symbols).
    (
        "use clause",
        "      use modal_aero_wateruptake, only: modal_aero_wateruptake_dr\n",
        "after",
        USE_LINE,
    ),
    # Before each of the three microphysics calls + after the call settles
    (
        "calcsize_before",
        "! call calcsize\n",
        "before",
        dump_call("calcsize_before"),
    ),
    (
        "calcsize_after",
        "         write(lun,'(a,i7)') 'calcsize tend = 0 for all species'\n",
        "after",  # inserts after the `end if` that follows. We use end-if line below.
        None,  # filled in below to anchor on `end if`
    ),
    (
        "wateruptake_before",
        "! call wateruptake\n",
        "before",
        dump_call("wateruptake_before"),
    ),
    (
        "wateruptake_after",
        "      call unload_pbuf( pbuf, lchnk, ncol, &\n",
        "after",
        None,  # anchor on the continuation line below
    ),
    (
        "amicphys_before",
        "      call modal_aero_amicphys_intr(              &\n",
        "before",
        dump_call("amicphys_before"),
    ),
    (
        "amicphys_after",
        "         wetdens,            qaerwat              )\n",
        "after",
        dump_call("amicphys_after"),
    ),
]


def apply(path: Path) -> None:
    text = path.read_text()
    if "mam4_dump_state" in text:
        print(f"[apply_instrumentation] {path.name} already instrumented; no-op.")
        return

    lines = text.splitlines(keepends=True)
    new_lines = lines[:]

    # First the simple before/after-anchored insertions
    for desc, anchor, where, payload in ANCHORS:
        if payload is None:
            continue  # handled below
        try:
            idx = new_lines.index(anchor)
        except ValueError as exc:  # pragma: no cover
            raise RuntimeError(
                f"[apply_instrumentation] anchor not found for {desc}: {anchor!r}"
            ) from exc
        if where == "before":
            new_lines.insert(idx, payload)
        elif where == "after":
            new_lines.insert(idx + 1, payload)
        else:  # pragma: no cover
            raise ValueError(where)

    # Two anchors need slightly more care because their insertion site is
    # a line or two beyond the matched anchor line.
    # calcsize_after: insert AFTER the `end if` that closes the tend>0 branch,
    #   which is the line immediately following the
    #   "'calcsize tend = 0 for all species'" write.
    cs_anchor = "         write(lun,'(a,i7)') 'calcsize tend = 0 for all species'\n"
    cs_idx = new_lines.index(cs_anchor)
    # The next two lines are "      end if\n" then "\n"; we want to insert
    # after the `end if` (before the blank).
    end_if_idx = next(
        i for i in range(cs_idx + 1, cs_idx + 5)
        if new_lines[i].strip() == "end if"
    )
    new_lines.insert(end_if_idx + 1, dump_call("calcsize_after"))

    # wateruptake_after: anchor is the FIRST occurrence of the unload_pbuf
    # call after the wateruptake call. The actual call spans 2 lines; we
    # insert after the closing paren line.
    wu_call_idx = next(
        i for i, line in enumerate(new_lines)
        if line == "! call wateruptake\n"
    )
    # First unload_pbuf after wu_call_idx
    unload_start = next(
        i for i in range(wu_call_idx, len(new_lines))
        if new_lines[i] == "      call unload_pbuf( pbuf, lchnk, ncol, &\n"
    )
    # Continuation line follows; closing of call is at unload_start + 1.
    new_lines.insert(unload_start + 2, dump_call("wateruptake_after"))

    path.write_text("".join(new_lines))
    print(f"[apply_instrumentation] patched {path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: apply_instrumentation.py <path-to-driver.F90>")
    apply(Path(sys.argv[1]))
