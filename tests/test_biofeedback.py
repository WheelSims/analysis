"""
Integration tests for the biofeedback processing and kinematic analysis module.

This module validates the integration and correct computation of real-time
wheelchair propulsion metrics using pre-recorded OptiTrack data simulated
in an offline environment.
"""

import os
import sys
import pytest

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "src"))
sys.path.append(root_dir)

import kineticstoolkit as ktk

from src import biofeedback


# %% Integration tests public functions
def test_with_udp_data_from_motive():
    """
    Validate the biofeedback update loop using prerecorded OptiTrack data.

    This test simulates an offline real-time stream by incrementally feeding
    time series segments into the "biofeedback_kinematics" pipeline. It verifies
    that the expected number of propulsion cycles and full kinematics time
    series samples are extracted. It also confirms that the detected propulsion
    cycles are classified with the expected push pattern labels, ensuring that
    the pattern recognition pipeline produces consistent results.
    """
    data_path = os.path.join(
        root_dir, "tests", "data", "optitrack_fetch.ktk.zip"
    )
    data = ktk.load(data_path)

    arg_path = os.path.join(
        root_dir,
        "tests",
        "data",
        "godot_argument_for_biofeedback_test.ktk.zip",
    )
    arg = ktk.load(arg_path)

    biofeedback._runtime_state["run_mode"] = "offline"

    nb_index = min(
        [
            len(data["102"].time),
            len(data["201"].time),
            len(data["202"].time),
        ]
    )
    data_temp = data.copy()

    for i in range(50, nb_index, 40):
        data_temp["102"] = data["102"].get_ts_before_index(i)
        data_temp["201"] = data["201"].get_ts_before_index(i)
        data_temp["202"] = data["202"].get_ts_before_index(i)
        biofeedback._runtime_state["data"] = data_temp
        biofeedback.biofeedback_kinematics(**arg)
    results = biofeedback.kinematics_data.copy()
    biofeedback.biofeedback_stop()

    # Verify the number of detected cycles and the full time series length
    y_1 = [
        len(results["cycles"]["left"]),
        len(results["cycles"]["right"]),
        len(results["ts_full"]["left"].time),
        len(results["ts_full"]["right"].time),
    ]

    assert [
        8,
        8,
        1224,
        1224,
    ] == y_1, f"TEST FAILED: Exp: [8, 8, 1224, 1224] | Got: {y_1} | "

    # Verify the push pattern labels of the detected cycles
    y_2 = [
        results["cycles"]["left"][0]["label_push_pattern"],
        results["cycles"]["left"][1]["label_push_pattern"],
        results["cycles"]["left"][2]["label_push_pattern"],
        results["cycles"]["left"][3]["label_push_pattern"],
        results["cycles"]["left"][4]["label_push_pattern"],
        results["cycles"]["left"][5]["label_push_pattern"],
        results["cycles"]["left"][6]["label_push_pattern"],
        results["cycles"]["left"][7]["label_push_pattern"],
    ]

    assert [
        "Pumping (PM)",
        "Pumping (PM)",
        "Single-Loop (SLOP)",
        "Single-Loop (SLOP)",
        "Semi-Circular (SC)",
        "Semi-Circular (SC)",
        "Double-Loop (DLOP)",
        "Double-Loop (DLOP)",
    ] == y_2, f"TEST FAILED: Exp: Pumping (PM) | Got: {y_2}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
