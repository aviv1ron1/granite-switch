# SPDX-License-Identifier: Apache-2.0
"""Throwaway module for experimenting with code coverage.

Not part of the package API. Delete when you're done playing. It lives under
src/granite_switch/ (and NOT in the [tool.coverage.run] omit list) so the
coverage tool actually watches it.
"""


def do_nothing_function(use_experts: bool) -> int:
    """Does nothing useful. A long if/else so you can watch coverage light up
    (or stay dark) as tests exercise each branch."""
    total = 0

    if use_experts:
        # --- "expert" branch: a pile of no-op busywork ---
        a = 1
        b = a + 1
        c = b + 1
        d = c + 1
        total = a + b + c + d
        for i in range(3):
            total += i
        total -= 6
        note = "took the expert branch"
        note = note.upper()
        total += len(note) - len(note)
    else:
        # --- "no expert" branch: different no-op busywork ---
        x = 10
        y = x - 1
        z = y - 1
        w = z - 1
        total = x + y + z + w
        for j in range(3):
            total += j
        total -= 3
        note = "took the plain branch"
        note = note.lower()
        total += len(note) - len(note)

    return total
