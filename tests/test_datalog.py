"""
Template for pytest.

Copy this file and change modulename for the module to be tested, then
modify test_functionname and maybe add other test functions.

"""

import os
import sys
import shutil
import time
from datetime import date, datetime

import kineticstoolkit as ktk
import numpy as np
import pandas as pd


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "src"))
sys.path.append(root_dir)

import data_logging

arg = {
    "folder": os.getcwd(),
    "participant": "unittest",
    "time": str(time.time()),
    "scene": "scene",
    "player_trajectory": True,
    "instrumented_wheels": False,
    "motion_capture": False,
    "position": "(0,0,0)",
    "rotation": "(0,0,0)",
}

trajectory = {
    "file": "trajectory",
    "headers": ["position", "rotation"],
    "columns": [4, 4],
}

wheels = {
    "file": "right",
    "headers": ["Analog", "IMU", "Encoder", "Power"],
    "sample": root_dir + "/tests/data/nextwheel_fetch.ktk.zip",
    "data": None,
}

motion = {
    "file": "rigidbody",
    "headers": ["102", "201", "202"],
    "sample": root_dir + "/tests/data/optitrack_untransformed.ktk.zip",
    "data": None,
}


# %% Initiation


def test_start_log():
    """
    Test start_log.

    Asserts folders were created for the test participant and today's session.

    """
    session_details = data_logging.FileDetails()

    data_logging.start_log(**arg, session_details=session_details)

    assert session_details.session_date == str(date.today()), (
        f"TEST start_trial: Session date {session_details.session_date} is incorrect."
    )

    assert isinstance(session_details.session, int), (
        f"TEST start_log: Session number {session_details.session} must be an integer."
    )

    assert os.path.isdir(session_details.participant_folder), (
        f"TEST start_trial: Folder {arg['participant']} was not created."
    )

    assert os.path.isdir(session_details.session_folder), (
        f"TEST start_trial: Session folder {str(date.today())} was not created."
    )

    if os.path.exists(os.path.join(arg["folder"], arg["participant"])):
        shutil.rmtree(os.path.join(arg["folder"], arg["participant"]))


def test_create_trial():
    """
    Test create_trial.

    Asserts that a new trial was initiated, and that the appropriate CSV file
    was created (to save the player trajectory).
    """
    session_details = data_logging.FileDetails()
    session_writers: dict[str, data_logging.FileLogger] = {}

    data_logging.start_log(**arg, session_details=session_details)
    data_logging.create_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )

    assert isinstance(session_details.trial, int), (
        f"TEST create_trial: Trial number {session_details.trial} must be an integer."
    )

    assert session_details.trial > 0, (
        f"TEST create_trial: Trial number {session_details.trial} must be positive."
    )

    assert os.path.isdir(session_details.trial_folder), (
        f"TEST create_trial: Trial folder {session_details.trial_folder} was not created."
    )

    assert os.path.isfile(
        os.path.join(
            session_details.trial_folder,
            session_writers["player_trajectory"].file.name.split("\\")[-1],
        )
    ), f"TEST create_trial: File {trajectory['file']} does not exist."

    writers = session_writers.copy()

    data_logging.end_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )
    data = pd.read_csv(
        os.path.join(
            session_details.trial_folder,
            writers["player_trajectory"].file.name.split("\\")[-1],
        ),
        sep=",",
    )

    assert list(data.columns) == ["time"] + [
        trajectory["headers"][i] + "[:," + str(j) + "]"
        for i in range(len(trajectory["headers"]))
        for j in range(trajectory["columns"][i])
    ], f"TEST create_trial: Headers for {trajectory['file']} are incorrect."

    if os.path.exists(os.path.join(arg["folder"], arg["participant"])):
        shutil.rmtree(os.path.join(arg["folder"], arg["participant"]))


# %% Ending


def test_end_trial():
    """
    Test end_trial.

    Assert files are saved and readable, and writers were erased.
    """
    session_details = data_logging.FileDetails()
    session_writers: dict[str, data_logging.FileLogger] = {}

    data_logging.start_log(**arg, session_details=session_details)
    data_logging.create_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )

    writers = session_writers.copy()

    data_logging.end_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )

    for key in session_writers.keys():
        assert os.path.isfile(
            os.path.join(
                session_details.trial_folder, writers[key]["filename"]
            )
        ), f"TEST end_trial: File {key} does not exist."

        file_writer = data_logging.FileLogger(
            os.path.join(
                session_details.trial_folder, writers[key]["filename"]
            )
        )
        assert file_writer.closed, f"File {key} is not closed."

    assert len(session_writers.keys()) == 0, (
        "TEST end_trial: Session writers still exist."
    )

    if os.path.exists(os.path.join(arg["folder"], arg["participant"])):
        shutil.rmtree(os.path.join(arg["folder"], arg["participant"]))


def test_end_log():
    """
    Test end_log.

    Assert files are saved and readable, and writers were erased even without
    a call to end_trial.
    """
    session_details = data_logging.FileDetails()
    session_writers: dict[str, data_logging.FileLogger] = {}

    data_logging.start_log(**arg, session_details=session_details)
    data_logging.create_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )

    writers = session_writers.copy()

    data_logging.end_log(
        **arg, session_details=session_details, session_writers=session_writers
    )

    for key in session_writers.keys():
        assert os.path.isfile(
            os.path.join(
                session_details.trial_folder, writers[key]["filename"]
            )
        ), f"TEST end_log: File {key} does not exist."

        file_writer = data_logging.FileLogger(
            os.path.join(
                session_details.trial_folder, writers[key]["filename"]
            )
        )
        assert file_writer.closed, f"File {key} is not closed."

    assert len(session_writers.keys()) == 0, (
        "TEST end_log: Session writers still exist."
    )

    if os.path.exists(os.path.join(arg["folder"], arg["participant"])):
        shutil.rmtree(os.path.join(arg["folder"], arg["participant"]))


# %% Saving


def test_save_data():
    """
    Test save_data.

    Assert that sample data for the player trajectory is properly saved, with
    the correct date.
    """
    session_details = data_logging.FileDetails()
    session_writers: dict[str, data_logging.FileLogger] = {}

    data_logging.start_log(**arg, session_details=session_details)
    data_logging.create_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )

    data_logging.save_data(
        **arg, session_details=session_details, session_writers=session_writers
    )
    writers = session_writers.copy()

    data_logging.end_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )
    data = pd.read_csv(
        os.path.join(
            session_details.trial_folder,
            writers["player_trajectory"].file.name.split("\\")[-1],
        ),
        sep=",",
    )

    for col in data.columns:
        if col == "position[:,3]":
            assert data.loc[0][col] == 1.0, (
                "TEST save_data: Third position column should hold value 1.0."
            )
        elif col == "rotation[:,3]":
            assert data.loc[0][col] == 0.0, (
                "TEST save_data: Third rotation column should hold value 0.0."
            )
        elif col == "time":
            assert (
                datetime.fromtimestamp(data.loc[0][col]).date() == date.today()
            ), "TEST save_data: Time column holds a value that are not today."
        else:
            assert isinstance(data.loc[0][col], float), (
                f"TEST save_data: Col {col} holds a value that is not a float."
            )

    if os.path.exists(os.path.join(arg["folder"], arg["participant"])):
        shutil.rmtree(os.path.join(arg["folder"], arg["participant"]))


def test_save_ts():
    """
    Test _save_ts for NextWheels and for Optitrack.

    Asserts that sample data from the instrumented wheels is properly saved,
    with the correct date.
    """
    session_details = data_logging.FileDetails()
    session_writers: dict[str, data_logging.FileLogger] = {}

    data_logging.start_log(**arg, session_details=session_details)
    data_logging.create_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )

    samples = [wheels, motion]
    for sample in samples:
        if arg["folder"].split("\\")[-1] == "tests":
            sample["data"] = ktk.load(
                os.path.join(arg["folder"], "data", sample["sample"])
            )
        else:
            sample["data"] = ktk.load(
                os.path.join(arg["folder"], "tests", "data", sample["sample"])
            )

        for file_type in sample["headers"]:
            filename = data_logging._make_filename(
                arg["scene"],
                sample["file"] + "_" + file_type,
                session_details=session_details,
            )

            data_logging._save_ts(
                sample["data"][file_type],
                sample["file"] + "_" + file_type,
                arg["scene"],
                session_writers=session_writers,
                session_details=session_details,
            )

            assert os.path.isfile(
                os.path.join(session_details.trial_folder, filename),
            ), f"TEST _save_ts: File {filename} is missing."

    data_logging.end_trial(
        **arg, session_details=session_details, session_writers=session_writers
    )

    for sample in samples:
        for file_type in sample["headers"]:
            filename = data_logging._make_filename(
                arg["scene"],
                sample["file"] + "_" + file_type,
                session_details=session_details,
            )

            data = pd.read_csv(
                os.path.join(session_details.trial_folder, filename), sep=","
            )

            data_header = list(
                sample["data"][file_type].to_dataframe().columns
            )

            assert list(data.columns)[1:] == data_header, (
                f"TEST _save_ts: Headers for {file_type} are incorrect."
            )

            assert np.allclose(sample["data"][file_type].time, data["time"]), (
                f"TEST _save_ts: Wheel data {file_type} is missing timestamps."
            )

            assert np.allclose(
                sample["data"][file_type].to_dataframe(),
                data.drop(columns="time"),
            ), f"TEST _save_ts: Data {file_type} not saved properly."

    if os.path.exists(os.path.join(arg["folder"], arg["participant"])):
        shutil.rmtree(os.path.join(arg["folder"], arg["participant"]))


if __name__ == "__main__":  # pragma: no cover
    import pytest

    pytest.main([__file__])
