#!/usr/bin/env python3
"""Pre-commit helper: bump patch version in pyproject.toml (0.1.2 -> 0.1.3)."""

from __future__ import annotations

import contextlib
import re
import subprocess
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# only match version inside [project] section to avoid bumping dependencies
# supports both " and ' quotes (defensive)
_PROJECT_VERSION_RE = re.compile(r"^version\s*=\s*[\"'](\d+)\.(\d+)\.(\d+)[\"']", re.MULTILINE)


def _find_project_version(text: str) -> tuple[re.Match[str] | None, str, str]:
    """Return (match, before, project_section) for version inside [project]."""
    if "[project]" not in text:
        return None, text, ""
    before, rest = text.split("[project]", 1)
    # project section ends at next top-level header [xxx]
    # find next "\n[" after rest
    end = rest.find("\n[")
    if end == -1:
        section = rest
        after = ""
    else:
        section = rest[:end]
        after = rest[end:]
    m = _PROJECT_VERSION_RE.search(section)
    return m, before, section + after  # keep after for reconstruction


def bump() -> bool:
    # Only bump when there's a staged commit (avoid bump on `pre-commit run --all-files`)
    with contextlib.suppress(Exception):
        # git diff --cached --quiet returns 0 if no staged changes
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        if result.returncode == 0 and "--force" not in sys.argv:
            # No staged changes -> not a real commit, skip bump
            return False
    text = PYPROJECT.read_text(encoding="utf-8")
    if "[project]" not in text:
        print("bump_patch: no [project] found", file=sys.stderr)
        return False
    # isolate [project] section
    before, rest = text.split("[project]", 1)
    # find next section
    next_header = rest.find("\n[")
    if next_header == -1:
        project_section = rest
        after = ""
    else:
        project_section = rest[:next_header]
        after = rest[next_header:]
    m = _PROJECT_VERSION_RE.search(project_section)
    if not m:
        print("bump_patch: no version found in [project]", file=sys.stderr)
        return False
    major, minor, patch = map(int, m.groups())
    new_version = f"{major}.{minor}.{patch + 1}"
    new_section = _PROJECT_VERSION_RE.sub(f'version = "{new_version}"', project_section, count=1)
    if new_section == project_section:
        return False
    new_text = before + "[project]" + new_section + after
    PYPROJECT.write_text(new_text, encoding="utf-8")
    print(f'bump_patch: {m.group(0).strip()} -> version = "{new_version}"')
    # re-stage file so commit includes the bump
    with contextlib.suppress(Exception):
        subprocess.run(["git", "add", str(PYPROJECT)], check=False)
    return True


if __name__ == "__main__":
    bump()
