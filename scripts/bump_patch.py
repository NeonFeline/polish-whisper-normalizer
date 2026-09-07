#!/usr/bin/env python3
"""Pre-commit helper: bump patch version in pyproject.toml (0.1.2 -> 0.1.3)."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

pattern = re.compile(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', re.MULTILINE)


def bump() -> bool:
    # Only bump when there's a staged commit (avoid bump on `pre-commit run --all-files`)
    try:
        # git diff --cached --quiet returns 0 if no staged changes
        result = subprocess.run(["git", "diff", "--cached", "--quiet"], check=False)
        if result.returncode == 0:
            # No staged changes -> not a real commit, skip bump
            # Allow manual `python scripts/bump_patch.py` to still bump when passed --force
            if "--force" not in sys.argv:
                return False
    except Exception:
        pass
    text = PYPROJECT.read_text(encoding="utf-8")
    m = pattern.search(text)
    if not m:
        print("bump_patch: no version found", file=sys.stderr)
        return False
    major, minor, patch = map(int, m.groups())
    new_version = f"{major}.{minor}.{patch + 1}"
    new_text = pattern.sub(f'version = "{new_version}"', text, count=1)
    if new_text == text:
        return False
    PYPROJECT.write_text(new_text, encoding="utf-8")
    print(f'bump_patch: {m.group(0).strip()} -> version = "{new_version}"')
    # re-stage file so commit includes the bump
    try:
        subprocess.run(["git", "add", str(PYPROJECT)], check=False)
    except Exception:
        pass
    return True


if __name__ == "__main__":
    bump()
