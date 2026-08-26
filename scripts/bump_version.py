#!/usr/bin/env python3
"""Bump the package version in the two places it lives.

The version has exactly two homes:

* ``mam4_jax/__init__.py`` -- ``__version__``, the single source of truth
  (``pyproject.toml`` reads it via ``version = {attr = ...}``, so there is
  deliberately no literal there to update).
* ``CHANGELOG.md`` -- the dated section heading.

Keeping those two in step by hand is what has gone wrong here before: ``v0.3.1``
was tagged from a tree whose ``__version__`` still said ``0.3.0``, and
``CHANGELOG.md`` claimed ``v0.3.2 -- unreleased`` for five days after v0.3.2 was
on PyPI. The tag ruleset blocks deletion, so each mistake burns a version
number permanently.

Usage
-----
    python scripts/bump_version.py 0.4.1        # explicit
    python scripts/bump_version.py minor        # 0.4.1 -> 0.5.0
    python scripts/bump_version.py patch --dry-run

This writes the version and opens the changelog section. It does NOT write the
changelog body, commit, or tag -- those stay deliberate.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INIT = ROOT / "mam4_jax" / "__init__.py"
CHANGELOG = ROOT / "CHANGELOG.md"

_VERSION_RE = re.compile(r'^__version__ = "([^"]+)"$', re.M)
_SEMVER_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def read_version() -> str:
    m = _VERSION_RE.search(INIT.read_text())
    if m is None:
        sys.exit(f"no `__version__ = \"...\"` line found in {INIT}")
    return m.group(1)


def parse(version: str) -> tuple[int, int, int]:
    m = _SEMVER_RE.match(version)
    if m is None:
        sys.exit(f"not a MAJOR.MINOR.PATCH version: {version!r}")
    return tuple(int(p) for p in m.groups())  # type: ignore[return-value]


def resolve_target(current: str, spec: str) -> str:
    """`spec` is either a literal version or one of major/minor/patch."""
    major, minor, patch = parse(current)
    if spec == "major":
        return f"{major + 1}.0.0"
    if spec == "minor":
        return f"{major}.{minor + 1}.0"
    if spec == "patch":
        return f"{major}.{minor}.{patch + 1}"
    parse(spec)  # validates, exits on malformed
    return spec


def git(*args: str) -> str | None:
    """Run a git command; None if git or the repo is unavailable."""
    try:
        out = subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip()


def check_preconditions(current: str, target: str) -> None:
    if parse(target) <= parse(current):
        sys.exit(f"refusing to bump {current} -> {target}: not an increase")

    tags = git("tag", "--list", f"v{target}")
    if tags:
        sys.exit(
            f"refusing: tag v{target} already exists. The tag ruleset blocks "
            f"deletion, so this number is spent -- pick the next one."
        )

    dirty = git("status", "--porcelain")
    if dirty:
        print("warning: working tree is not clean:\n" + dirty, file=sys.stderr)

    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch is not None and branch not in ("main", "HEAD"):
        print(
            f"warning: on branch {branch!r}. Release commits and tags belong on "
            f"a branch cut from main -- a tag on any other lineage ships "
            f"whatever that branch contains.",
            file=sys.stderr,
        )


def bump_init(target: str, dry_run: bool) -> None:
    text = INIT.read_text()
    new = _VERSION_RE.sub(f'__version__ = "{target}"', text, count=1)
    if new == text:
        sys.exit(f"{INIT}: __version__ line did not change")
    if not dry_run:
        INIT.write_text(new)
    print(f"  {INIT.relative_to(ROOT)}: __version__ = \"{target}\"")


def bump_changelog(target: str, today: str, dry_run: bool) -> None:
    text = CHANGELOG.read_text()
    heading = f"## v{target}"
    existing = re.search(rf"^{re.escape(heading)}\s*(?:--|—)?\s*(.*)$", text, re.M)

    if existing is not None:
        # Section already drafted (commonly "-- unreleased"): date it.
        old_line = existing.group(0)
        new_line = f"{heading} — {today}"
        if old_line == new_line:
            print(f"  {CHANGELOG.name}: section for v{target} already dated")
            return
        text = text.replace(old_line, new_line, 1)
        print(f"  {CHANGELOG.name}: dated existing section -> {new_line!r}")
    else:
        anchor = "# Changelog\n\n"
        if anchor not in text:
            sys.exit(f"{CHANGELOG}: no '# Changelog' header to insert under")
        section = (
            f"{anchor}{heading} — {today}\n\n"
            "<!-- Write the release notes here: Added / Changed / Fixed. -->\n"
            "<!-- Call out behavioural changes explicitly; hosts read this. -->\n\n"
        )
        text = text.replace(anchor, section, 1)
        print(f"  {CHANGELOG.name}: inserted section '{heading} — {today}'")

    if not dry_run:
        CHANGELOG.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="explicit version (0.4.1) or major/minor/patch")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would change without writing")
    ap.add_argument("--date", default=None,
                    help="release date for the changelog heading (default: today)")
    args = ap.parse_args()

    current = read_version()
    target = resolve_target(current, args.spec)
    today = args.date or _dt.date.today().isoformat()

    check_preconditions(current, target)

    print(f"{current} -> {target}" + ("  (dry run)" if args.dry_run else ""))
    bump_init(target, args.dry_run)
    bump_changelog(target, today, args.dry_run)

    if args.dry_run:
        return
    print(
        "\nNext, in order:\n"
        f"  1. Write the v{target} notes in CHANGELOG.md.\n"
        "  2. python -m pytest -q\n"
        "  3. PR to main; merge it.\n"
        f"  4. Verify __version__ on the MERGED commit, then:\n"
        f"     git tag v{target} <merge-sha> && git push origin v{target}\n"
        "     Pushing the tag IS the PyPI release (.github/workflows/publish.yml).\n"
    )


if __name__ == "__main__":
    main()
