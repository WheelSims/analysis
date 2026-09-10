"""
Run the commands related to data logging called from the python bridge.

Data logged includes the player's virtual position in the simulator, as
well as information collected through the instrumented wheels.

"""

import glob
import os
from datetime import date

import kineticstoolkit as ktk
import numpy as np
from nextwheel import NextWheel

import optitrack as ot

# %% Session dictionaries

wheels = {
    "wheels": {
        "right": NextWheel(),
        # "left": NextWheel(),
    },
    "ip_addresses": {
        "right": "192.168.0.86",
        "left": "192.168.0.13",
    },
}


# %% Classes to hold data and arguments


class FileLogger:
    """FileLogger class holds an open file of filename."""

    def __init__(self, filename):
        """Initialize FileLogger object."""
        self.file = open(
            filename, "w", encoding="utf-8", buffering=1024 * 1024
        )

    def log_row(self, data_lines: np.ndarray) -> None:
        """
        Log data to self.file.

        Parameters
        ----------
        data_lines :
            Data to append to the file.

        Raises
        ------
        ValueError
            Error is raised when trying to write to a closed file.
        """
        if self.file and not self.file.closed:
            for row in data_lines:
                self.file.write(",".join(map(str, row)) + "\n")

        else:
            raise ValueError(f"File {self.file.name('\\')[-1]} is closed.")

    def close_log(self) -> None:
        """Close an opened file."""
        if self.file:
            self.file.close()


session_writers: dict[str, FileLogger] = {}


class FileDetails:
    """FileDetails class holds information about current session/trial."""

    def __init__(
        self,
        participant_folder: str = "",
        session: int = 0,
        session_date: str = "",
        session_folder: str = "",
    ):
        """Initialize FileDetails object."""
        self.participant_folder = participant_folder
        self.session = session
        self.session_date = session_date
        self.session_folder = session_folder
        self.trial = 0
        self.trial_folder = ""


session_details = FileDetails()


# %% Folder contents


def _make_folder(
    directory: str,
    participant: str,
    session: str = "",
    trial: str = "",
) -> str:
    """
    Create a folder for a specific paricipant within directory.

    Sub-folders for sessions and trials can be created through this function
    when those arguments are included.

    Parameters
    ----------
    directory
        Base folder containing data for all participants.
    participant
        Participant identifier number.
    session
        Optional. Current session number.
    trial
        Optional. Current trial number.

    Returns
    -------
    str
        Folder specific to this participant (and/or session and trial).

    """
    folder = os.path.join(directory, participant, session, trial)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print("Created folder ", folder)
    return folder


def _get_number(folder: str) -> int:
    """
    Identify session or trial currently in-progress through folder-parsing.

    If none are found, returns 0.

    Parameters
    ----------
    folder
        Folder corresponding to current participant and/or session.

    Returns
    -------
    str
        Current session number.

    """
    folders = [
        f for f in glob.glob(os.path.join(folder, "*")) if os.path.isdir(f)
    ]
    if (len(folders)) > 0:
        number = len(folders)
    else:
        number = 0
    return number


# %% File generation


def _make_filename(
    scene: str,
    data_type: str,
    session_details: FileDetails = session_details,
) -> str:
    """
    Create a filename appropriate for the trajectory data to be saved.

    Parameters
    ----------
    scene :
        Current playable scene selected (out of 6 options).
    data_type :
        The type of data to be saved from Simulator or instrumented wheels.
        Options are:
            Simulator (through Godot): trajectory.
            Optitrack: RigidBody + ID.
            NextWheel: Analog, IMU, Encoder, Power.
    session_details :
        FileDetails() object holding details for current session/trial.
        The default is session_details.

    Returns
    -------
    str
        Name of file to be created.

    """
    file = (
        "S"
        + str(session_details.session)
        + "_"
        + session_details.session_date
        + "_"
        + "T"
        + str(session_details.trial)
        + "_"
        + scene
        + "_"
        + data_type
        + ".csv"
    )

    return file


def _make_file(
    filename: str,
    header: list[list[str]],
    filetype: str,
    session_writers: dict[str, FileLogger] = session_writers,
) -> None:
    """
    Create a file of a particular header within specified folder.

    Parameters
    ----------
    filename :
        Name of file to be created.
    header :
        Header of file to be created.
    filetype :
        Type of file to be created, used as key in session_writers.
    session_writers :
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.

    """
    session_writers[filetype] = FileLogger(filename)
    session_writers[filetype].log_row(np.array(header))


# %% Logging simulator


def _make_header(
    data_headers: list[str],
    data_columns: list[int],
) -> list[list[str]]:
    """
    Create a header appropriate for the type of data to be saved.

    Parameters
    ----------
    data_headers
        The specific column titles to be saved in the file.
        For player_trajectory, the input should be ['position', 'rotation'].
    data_columns
        The number of columns per column title.
        For player_trajectory, the input should be [4, 4].

    Returns
    -------
    list[str]
        Header to be used when creating the file.

    """
    header = [
        ['"time"']
        + [
            '"' + data_headers[i] + "[:," + str(j) + ']"'
            for i in range(len(data_headers))
            for j in range(data_columns[i])
        ]
    ]
    return header


def _save_trajectory(
    data_values: dict[str, str],
    session_writers: dict[str, FileLogger] = session_writers,
) -> None:
    """
    Append data to an open file containing trajectory.

    Parameters
    ----------
    data_values :
        Current data values to save.
    session_writers :
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.

    """
    data_line = [
        [data_values["time"]]
        + list(data_values["position"].strip("()").split(","))
        + ["1"]
        + list(data_values["rotation"].strip("()").split(","))
        + ["0"]
    ]

    session_writers["player_trajectory"].log_row(np.array(data_line))


# %% Logging TimeSeries


def _save_ts(
    ts: ktk.TimeSeries,
    filetype: str,
    scene: str,
    session_writers: dict[str, FileLogger] = session_writers,
    session_details: FileDetails = session_details,
) -> None:
    """
    Open and append data to open file containing time series data.

    Parameters
    ----------
    ts :
        Newly-fetched data from NextWheel or Optitrack.
    filetype :
        Type of file to be created, used as key in session_writers.
    scene :
        Current scene.
    session_writers :
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.
    session_details :
        FileDetails() object holding details for current session/trial.
        The default is session_details.

    """
    if len(ts.time) > 0:
        if filetype not in session_writers:
            header = [
                ['"time"'] + [f'"{s}"' for s in ts.to_dataframe().columns]
            ]
            filename = _make_filename(scene, filetype, session_details)
            _make_file(
                os.path.join(str(session_details.trial_folder), filename),
                header,
                filetype,
                session_writers,
            )

        data_lines = np.column_stack(
            [
                ts.data[list(ts.data.keys())[i]]
                for i in range(len(list(ts.data.keys())))
            ]
        )

        session_writers[filetype].log_row(
            np.column_stack(
                (
                    ts.time,
                    data_lines,
                )
            )
        )


# %% Stopping devices external to Simulator


def _stop_wheels(
    scene: str,
    session_writers: dict[str, FileLogger] = session_writers,
    session_details: FileDetails = session_details,
) -> None:
    """
    Stop instrumented wheels streaming and catch final events.

    Parameters
    ----------
    scene :
        Current scene.
    session_writers :
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.
    session_details :
        FileDetails() object holding details for current session/trial.
        The default is session_details.

    """
    for key, wheel in wheels["wheels"].items():
        wheel.stop_streaming()
        print("Successfully stopped stream from wheel: " + wheel.ip)
        nw = wheel.fetch(clear=True)
        for subkey, ts in nw.items():
            _save_ts(
                ts,
                key + "_" + subkey,
                "instrumented_wheels",
                session_writers,
                session_details,
            )


def _stop_ot(
    scene: str,
    session_writers: dict[str, FileLogger] = session_writers,
    session_details: FileDetails = session_details,
):
    """
    Stop Optitrack streaming and catch final events.

    Parameters
    ----------
    scene:
        Current scene.
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.
    session_details :
        Dictionary holding folder details for current session/trial.
        The default is session_details.

    """
    motion = ot.fetch(clear_buffer=True, transform_data=False)
    ot.stop()
    print("Streaming ended for optitrack.")
    for ID, ts in motion.items():
        if len(ID) == 3:
            _save_ts(
                ts, "rigidbody_" + ID, scene, session_writers, session_details
            )


# %% Public functions


def start_log(
    *,
    folder: str,
    participant: str,
    time: str,
    scene: str,
    player_trajectory: bool,
    instrumented_wheels: bool,
    motion_capture: bool,
    position: str,
    rotation: str,
    session_details: FileDetails = session_details,
) -> None:
    """
    Create folders for current (new) session, in which trials will be saved.

    Parameters
    ----------
    folder:
        The main folder where all data is saved.
    participant:
        The current participant identifier.
    time:
        The current timestamp.
    scene:
        The current selected playable scene.
    player_trajectory:
        Whether to save the player's trajectory.
    instrumented_wheels:
        Whether to save the wheels.
    motion_capture:
        Whether to save the motion capture.
    position:
        The current player position in the simulator.
    rotation:
        The current player rotation in the simulator.
    session_details :
        FileDetails() object holding details for current session/trial.
        The default is session_details.

    """
    session_details.session_date = str(date.today())
    session_details.participant_folder = _make_folder(folder, participant)
    session_details.session_folder = _make_folder(
        folder,
        participant,
        session=session_details.session_date,
    )
    session_details.session = _get_number(session_details.participant_folder)

    if instrumented_wheels:
        for key, wheel in wheels["wheels"].items():
            try:
                wheel.ip = wheels["ip_addresses"][key]
                print(
                    "Successfully established connection to wheel: " + wheel.ip
                )
            except TimeoutError:
                print(
                    "Connection could not be established to wheel: " + wheel.ip
                )


def create_trial(
    *,
    folder: str,
    participant: str,
    time: str,
    scene: str,
    player_trajectory: bool,
    instrumented_wheels: bool,
    motion_capture: bool,
    position: str,
    rotation: str,
    session_writers: dict[str, FileLogger] = session_writers,
    session_details: FileDetails = session_details,
) -> None:
    """
    Create empty files where data will be saved during this current trial.

    Parameters
    ----------
    folder:
        The main folder where all data is saved.
    participant:
        The current participant identifier.
    time:
        The current timestamp.
    scene:
        The current selected playable scene.
    player_trajectory:
        Whether to save the player's trajectory.
    instrumented_wheels:
        Whether to save the wheels.
    motion_capture:
        Whether to save the motion capture.
    position:
        The current player position in the simulator.
    rotation:
        The current player rotation in the simulator.
    session_writers :
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.
    session_details :
        FileDetails() object holding details for current session/trial.
        The default is session_details.

    """
    if session_details.session_date is None:
        start_log(
            folder=folder,
            participant=participant,
            time=time,
            scene=scene,
            player_trajectory=player_trajectory,
            instrumented_wheels=instrumented_wheels,
            motion_capture=motion_capture,
            position=position,
            rotation=rotation,
            session_details=session_details,
        )

    if instrumented_wheels:
        for key, wheel in wheels["wheels"].items():
            wheel.start_streaming()
            print("Streaming started for wheel: ", key, " of IP ", wheel.ip)

    if motion_capture:
        ot.start()
        print("Streaming started for Optitrack.")

    session_details.trial = _get_number(session_details.session_folder) + 1

    session_details.trial_folder = _make_folder(
        folder,
        participant,
        session=session_details.session_date,
        trial="T" + str(session_details.trial),
    )

    if player_trajectory:
        filename = _make_filename(
            scene,
            "trajectory",
            session_details,
        )

        header = _make_header(["position", "rotation"], [4, 4])
        _make_file(
            os.path.join(session_details.trial_folder, filename),
            header,
            "player_trajectory",
            session_writers,
        )
        print("Created the file " + filename)


def save_data(
    *,
    folder: str,
    participant: str,
    time: str,
    scene: str,
    player_trajectory: bool,
    instrumented_wheels: bool,
    motion_capture: bool,
    position: str,
    rotation: str,
    session_writers: dict[str, FileLogger] = session_writers,
    session_details: FileDetails = session_details,
) -> None:
    """
    Open and append new data line to trajectory and instrumented wheels files.

    Parameters
    ----------
    folder:
        The main folder where all data is saved.
    participant:
        The current participant identifier.
    time:
        The current timestamp.
    scene:
        The current selected playable scene.
    player_trajectory:
        Whether to save the player's trajectory.
    instrumented_wheels:
        Whether to save the wheels.
    motion_capture:
        Whether to save the motion capture.
    position:
        The current player position in the simulator.
    rotation:
        The current player rotation in the simulator.
    session_writers :
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.
    session_details :
        FileDetails() object holding details for current session/trial.
        The default is session_details.

    """
    if not session_writers:
        create_trial(
            folder=folder,
            participant=participant,
            time=time,
            scene=scene,
            player_trajectory=player_trajectory,
            instrumented_wheels=instrumented_wheels,
            motion_capture=motion_capture,
            position=position,
            rotation=rotation,
            session_writers=session_writers,
            session_details=session_details,
        )

    if player_trajectory:
        _save_trajectory(
            {"time": time, "position": position, "rotation": rotation},
            session_writers,
        )

    if instrumented_wheels:
        for key, wheel in wheels["wheels"].items():
            nw = wheel.fetch(clear=True)
            for subkey, ts in nw.items():
                _save_ts(
                    ts,
                    key + "_" + subkey,
                    scene,
                    session_writers,
                    session_details,
                )

    if motion_capture:
        motion = ot.fetch(clear_buffer=True, transform_data=False)
        for ID, ts in motion.items():
            if len(ID) == 3:
                _save_ts(
                    ts,
                    "rigidbody_" + ID,
                    scene,
                    session_writers,
                    session_details,
                )


def end_trial(
    *,
    folder: str,
    participant: str,
    time: str,
    scene: str,
    player_trajectory: bool,
    instrumented_wheels: bool,
    motion_capture: bool,
    position: str,
    rotation: str,
    session_writers: dict[str, FileLogger] = session_writers,
    session_details: FileDetails = session_details,
) -> None:
    """
    Confirm the end of recording and terminate instrumented wheels streaming.

    Parameters
    ----------
    folder:
        The main folder where all data is saved.
    participant:
        The current participant identifier.
    time:
        The current timestamp.
    scene:
        The current selected playable scene.
    player_trajectory:
        Whether to save the player's trajectory.
    instrumented_wheels:
        Whether to save the wheels.
    motion_capture:
        Whether to save the motion capture.
    position:
        The current player position in the simulator.
    rotation:
        The current player rotation in the simulator.
    session_writers :
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.
    session_details :
        FileDetails() object holding details for current session/trial.
        The default is session_details.
    """
    if instrumented_wheels:
        _stop_wheels(
            scene,
            session_writers,
            session_details,
        )

    if motion_capture:
        _stop_ot(scene, session_writers, session_details)

    for _key, writer in session_writers.items():
        writer.close_log()

    session_writers.clear()

    print(
        "Logging is done for current trial: ",
        session_details.trial_folder,
    )


def end_log(
    *,
    folder: str,
    participant: str,
    time: str,
    scene: str,
    player_trajectory: bool,
    instrumented_wheels: bool,
    motion_capture: bool,
    position: str,
    rotation: str,
    session_writers: dict[str, FileLogger] = session_writers,
    session_details: FileDetails = session_details,
) -> None:
    """
    Ensure end_trial is called on the final trial.

    Parameters
    ----------
    folder:
        The main folder where all data is saved.
    participant:
        The current participant identifier.
    time:
        The current timestamp.
    scene:
        The current selected playable scene.
    player_trajectory:
        Whether to save the player's trajectory.
    instrumented_wheels:
        Whether to save the wheels.
    motion_capture:
        Whether to save the motion capture.
    position:
        The current player position in the simulator.
    rotation:
        The current player rotation in the simulator.
    session_writers :
        Dictionary holding all the FileLogger objects for this session.
        The default is session_writers.
    session_details :
        FileDetails() object holding details for current session/trial.
        The default is session_details.
    """
    if len(session_writers) > 0:
        end_trial(
            folder=folder,
            participant=participant,
            time=time,
            scene=scene,
            player_trajectory=player_trajectory,
            instrumented_wheels=instrumented_wheels,
            motion_capture=motion_capture,
            position=position,
            rotation=rotation,
            session_writers=session_writers,
            session_details=session_details,
        )
