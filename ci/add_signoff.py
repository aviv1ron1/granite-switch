# SPDX-License-Identifier: Apache-2.0

"""Append a DCO ``Signed-off-by`` trailer to the commit message if missing.

Runs at the ``prepare-commit-msg`` stage so contributors don't have to pass
``-s`` on every commit. The sign-off is derived from the Git author identity
(``user.name`` / ``user.email``), exactly like ``git commit -s`` does, so it
stays in sync with whatever the ``check-dco`` hook validates.
"""

import re
import subprocess
import sys
from pathlib import Path


def author_signoff() -> str | None:
    """Return ``Signed-off-by: Name <email>`` from the Git author identity."""
    try:
        ident = subprocess.check_output(
            ["git", "var", "GIT_AUTHOR_IDENT"], encoding="utf-8"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    # GIT_AUTHOR_IDENT looks like "Name <email> 1700000000 +0000"; keep up to '>'.
    match = re.match(r"^(.*<.+@.+>)", ident.strip())
    if not match:
        return None
    return f"Signed-off-by: {match.group(1)}"


def add_signoff(commit_msg_file: Path) -> bool:
    """Append the sign-off trailer if it isn't already present."""
    signoff = author_signoff()
    if signoff is None:
        print("⚠️  Could not determine Git author identity; skipping auto sign-off.")
        print("    Set it with: git config user.name / git config user.email")
        return False

    message = commit_msg_file.read_text(encoding="utf-8")

    # Ignore comment lines (git strips them) when checking for an existing trailer.
    body = "\n".join(
        line for line in message.splitlines() if not line.lstrip().startswith("#")
    )
    if signoff in body:
        return False

    # Ensure the trailer is separated from the body by a blank line.
    separator = "" if message.endswith("\n\n") else "\n" if message.endswith("\n") else "\n\n"
    commit_msg_file.write_text(f"{message}{separator}{signoff}\n", encoding="utf-8")
    print(f"✍️  Added DCO sign-off: {signoff}")
    return True


def main() -> int:
    """Entry point: expects the commit message file path as the first argument."""
    if len(sys.argv) < 2:
        print("Usage: add_signoff.py <commit-msg-file>")
        return 1

    add_signoff(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
