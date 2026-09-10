"""Test for biofeedback_pushrim_kinetics module."""

import matplotlib.pyplot as plt
import pytest
import os
import sys


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "src"))
sys.path.append(root_dir)


import biofeedback_pushrim_kinetics


def test_run_doesnt_crash():
    """Check that running the function won't crash."""
    biofeedback_pushrim_kinetics.connect("dummy")

    for i in range(10):
        biofeedback_pushrim_kinetics.process()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
