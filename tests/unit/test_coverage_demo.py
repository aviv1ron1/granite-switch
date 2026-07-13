# SPDX-License-Identifier: Apache-2.0
"""Coverage-demo tests. Throwaway — delete along with coverage_demo.py."""

from granite_switch.coverage_demo import do_nothing_function


def test_expert_branch():
    # Exercises ONLY the `if use_experts:` branch. The else: block never runs,
    # so coverage should report those lines as Missed.
    do_nothing_function(True)
