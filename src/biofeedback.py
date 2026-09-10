"""
Real-time wheelchair biofeedback processing and kinematic analysis module.

This module streams and processes raw tracking data from OptiTrack
(Motive software) in real time using a rolling window approach.

It filters local kinematics, extracts and validates voluntary propulsion
cycles, computes geometric metrics (A1, A2, areas), and classifies
push patterns.

The resulting metrics are synchronized and sent to the Godot engine.
"""

import time
from typing import Final, Literal, TypedDict

import kineticstoolkit as ktk
import matplotlib.pyplot as plt
import numpy as np

import optitrack as ot
import propulsion_patterns
from python_bridge import sender

# %% Public variables
# Current analysis window (n s).
LIMIT_DURATION_CURRENT_WINDOW = 5.0

# OptiTrack rigid body ID assigned to the wheelchair simulator
ID_WHEELCHAIR_SIMULATOR: Final = "102"

# OptiTrack rigid body ID assigned to the left hand marker cluster
ID_LEFT_HAND_CLUSTER: Final = "201"

# OptiTrack rigid body ID assigned to the right hand marker cluster
ID_RIGHT_HAND_CLUSTER: Final = "202"


# %% Typing classes
class Arg(TypedDict):
    """
    Configuration arguments for the biofeedback system.

    coordinates_left_wheel_center :
        3D calibration coordinates (X, Y, Z) for the left wheel center.
    coordinates_right_wheel_center :
        3D calibration coordinates (X, Y, Z) for the right wheel center.
    coordinates_left_hand :
        Initial 3D coordinates (X, Y, Z) of the marker on the left hand.
    coordinates_right_hand :
        Initial 3D coordinates (X, Y, Z) of the marker on the right hand.
    wheel_diameter :
        Diameter of the wheelchair wheels in meters.
    """

    coordinates_left_wheel_center: tuple[float, float, float]
    coordinates_right_wheel_center: tuple[float, float, float]
    coordinates_left_hand: tuple[float, float, float]
    coordinates_right_hand: tuple[float, float, float]
    wheel_diameter: float


class DataSide(TypedDict):
    """
    Side-specific streaming and calibration metadata.

    id_streaming :
        OptiTrack rigid body streaming ID unique to each side.
    local_meta2 :
        Local transformation for kinematic calculations.
    side :
        Label indicating the body side.
    wheel_center :
        3D position vector of the wheel center.
    """

    id_streaming: str
    local_meta2: np.ndarray
    side: Literal["left", "right"]
    wheel_center: np.ndarray


class Areas(TypedDict):
    """
    Signed geometric areas and trajectories between propulsion phases.

    sign :
        Sign of the deviation between recovery and push curves.
    area : float
        Calculated surface area between the segmented trajectories.
    recovery_phase :
        Array of coordinates representing the recovery path for this segment.
    push_phase :
        Array of coordinates representing the push path for this segment.
    """

    sign: Literal["positive", "negative"]
    area: float
    recovery_phase: np.ndarray
    push_phase: np.ndarray


class PushCycle(TypedDict):
    """
    Calculated metrics and kinematics for a single full propulsion cycle.

    in_push :
        Time and value marking the start of the push phase.
    recovery :
        Time and value marking the start of the recovery phase.
    end_push :
        Time and value marking the end of the push phase.
    range :
        Total linear displacement achieved along the anteroposterior axis
        during the push phase
    velocity_max :
        Maximum velocity reached during the cycle.
    push_frequency :
        Frequency of the propulsion cycle (Hz), calculated as the inverse of
        the cycle duration (1 / delta_t).
    normalised_push_pattern :
        Array of shape (101, 3) representing the resampled 3D hand trajectory
        normalized over 100% of the cycle.
    areas :
        List of dictionaries containing signed geometric areas between the push
        and recovery trajectories, or None if phases could not be segmented.
    A1 :
        Normalized recovery-phase deviation index. Compares hand deviation
        during recovery to a reference threshold. None if extraction fails.
    A2 :
        Symmetry index based on signed areas ranging from -1(negative
        dominance) to +1 (positive dominance). None if extraction fails.
    label_push_pattern :
        Classification label of the propulsion pattern based on geometric
        criteria (A1 and A2), or None if pattern classification was not
        possible.
    """

    in_push: dict[Literal["time", "value"], float]
    recovery: dict[Literal["time", "value"], float]
    end_push: dict[Literal["time", "value"], float]
    range: float
    velocity_max: float
    push_frequency: float
    normalised_push_pattern: np.ndarray
    areas: list[Areas] | None
    A1: float | None
    A2: float | None
    label_push_pattern: (
        Literal[
            "Pumping (PM)",
            "Semi-Circular (SC)",
            "Single-Loop (SLOP)",
            "Double-Loop (DLOP)",
            "",
        ]
        | None
    )


class KtkDataAndCycles(TypedDict):
    """
    Container for processed time-series kinematics and detected cycles.

    ts :
        Kineticstoolkit TimeSeries object containing side-specific kinematics.
    cycles :
        List of all valid propulsion cycles detected within the time window.
    """

    ts: ktk.TimeSeries | None
    cycles: list[PushCycle] | None


class RuntimeState(TypedDict):
    """
     Global state and accumulated processing results for biofeedback system.

     run_mode :
         Current execution state controlling real-time data acquisition.
         - 'start'  : Live streaming from OptiTrack.
         - 'stop'   : Idle state, acquisition stopped,structures are cleared.
         - 'offline': Test with injected local data.
     data :
         Raw input data mapping stream names to TimeSeries.
     current_window_data :
         Processed kinematic data and cycles scoped to the active time window.
    new_cycle_log :
         Counter or index tracking recently logged cycles for file writing.
     new_cycle_send :
         Counter or index tracking cycles sent to the biofeedback display.
    """

    run_mode: Literal["start", "stop", "offline"]
    data: dict[str, ktk.TimeSeries] | None
    current_window_data: (
        dict[Literal["left", "right"], KtkDataAndCycles] | None
    )
    new_cycle_log: dict[Literal["left", "right"], int]
    new_cycle_send: dict[Literal["left", "right"], int]


class KinematicsData(TypedDict):
    """
    Accumulated cycle metrics and full session time-series.

    cycles :
        Historical or cumulative list of push cycles for each side.
    ts_full :
        Full continuous time-series history aggregated since the session start.
    """

    cycles: dict[Literal["left", "right"], list]
    ts_full: dict[Literal["left", "right"], ktk.TimeSeries | None]


# %% Private variables
_runtime_state: RuntimeState = {
    "run_mode": "stop",
    "data": None,
    "current_window_data": None,
    "new_cycle_log": {"left": 1, "right": 1},
    "new_cycle_send": {"left": 3, "right": 3},
}

# Initialize storage for bilateral propulsion kinematic data
kinematics_data: KinematicsData = {
    "cycles": {"left": [], "right": []},
    "ts_full": {"left": None, "right": None},
}


# %% Public functions
def biofeedback_stop() -> None:
    """
    Clear all data.

    Stop the module optitrack and clear the ot data.
    Display the full kinematics and push pattern graphics
    (by default is commented)
    """
    try:
        # Display full kinematics and push pattern graphics at script
        # termination.
        # Reconstructs the global session dataset (limit_duration=0)
        # and injects the complete accumulated cycle history for both sides.

        if (
            kinematics_data["ts_full"]["left"] is not None
            and kinematics_data["ts_full"]["right"] is not None
        ):
            dict_ts_propulsion_cycles: dict[
                Literal["left", "right"], ktk.TimeSeries
            ] = {
                "left": kinematics_data["ts_full"]["left"],
                "right": kinematics_data["ts_full"]["right"],
            }

            dict_cycles: dict[Literal["left", "right"], list[PushCycle]] = {
                "left": kinematics_data["cycles"]["left"],
                "right": kinematics_data["cycles"]["right"],
            }

            propulsion_patterns.plot_bilateral_cycles(
                dict_ts_propulsion_cycles, dict_cycles
            )

            propulsion_patterns.plot_unilateral_push_patterns(
                kinematics_data["ts_full"]["left"],
                kinematics_data["cycles"]["left"],
            )

            propulsion_patterns.plot_unilateral_push_patterns(
                kinematics_data["ts_full"]["right"],
                kinematics_data["cycles"]["right"],
            )

    except Exception as e:
        print(f"Display full kinematics and push pattern : {e}")

    _init_variables()

    if _runtime_state["run_mode"] == "start":
        try:
            ot.stop()
            ot.clear()
        except Exception as e:
            print(f"close and clear optitrack streaming : {e}")

    print("Biofeedback closed")

    plt.show()


def biofeedback_update(
    coordinates_left_wheel_center: tuple[float, float, float],
    coordinates_right_wheel_center: tuple[float, float, float],
    coordinates_left_hand: tuple[float, float, float],
    coordinates_right_hand: tuple[float, float, float],
    wheel_diameter: float,
) -> None:
    """
    Execute an update iteration for the biofeedback (live and offline modes).

    Handles the streaming state machine: initializes OptiTrack on startup,
    fetches new tracking frames in 'start' mode, or processes pre-loaded local
    data in 'offline' mode. It extracts and filters kinematics, detects
    propulsion cycles, logs progress, and streams computed metrics to Godot.

    Parameters
    ----------
    coordinates_left_wheel_center :
        3D calibration coordinates (X, Y, Z) for the left wheel center.
    coordinates_right_wheel_center :
        3D calibration coordinates (X, Y, Z) for the right wheel center.
    coordinates_left_hand :
        Initial 3D coordinates (X, Y, Z) of the marker on the left hand.
    coordinates_right_hand :
        Initial 3D coordinates (X, Y, Z) of the marker on the right hand.
    wheel_diameter :
        Diameter of the wheelchair wheels in meters.

    """
    coordinates = Arg(
        {
            "coordinates_left_wheel_center": coordinates_left_wheel_center,
            "coordinates_right_wheel_center": coordinates_right_wheel_center,
            "coordinates_left_hand": coordinates_left_hand,
            "coordinates_right_hand": coordinates_right_hand,
            "wheel_diameter": wheel_diameter,
        }
    )

    if _runtime_state["run_mode"] == "stop":
        ot.start()
        time.sleep(1)
        print("Biofeedback started")
        _runtime_state["run_mode"] = "start"

    elif _runtime_state["run_mode"] == "start":
        try:
            _runtime_state["data"] = ot.fetch()
        except Exception as e:
            print(e)

        if not _runtime_state["data"]:
            return

        _execute_analysis_pipeline(
            _runtime_state,
            kinematics_data,
            coordinates,
            LIMIT_DURATION_CURRENT_WINDOW,
        )

    elif _runtime_state["run_mode"] == "offline":
        if not _runtime_state["data"]:
            return

        _execute_analysis_pipeline(
            _runtime_state,
            kinematics_data,
            coordinates,
            LIMIT_DURATION_CURRENT_WINDOW,
        )


# %% Private functions
def _analyze_current_window(
    data: dict[str, ktk.TimeSeries],
    arg: Arg,
    prev_data_cycles: dict[Literal["left", "right"], list],
    limit_duration: float = LIMIT_DURATION_CURRENT_WINDOW,
) -> dict[Literal["left", "right"], KtkDataAndCycles]:
    """
    Extract kinematics and validated propulsion cycles.

    Processing is limited to the current real-time data window.
    """
    # Initialize the current window data
    current_window_data: dict[Literal["left", "right"], KtkDataAndCycles] = {
        "left": {"ts": None, "cycles": None},
        "right": {"ts": None, "cycles": None},
    }

    data_side = _initialize_data_side(arg)
    data_windowed = _get_windowed_data(data, limit_duration)

    # Compute kinematics and cycles for left and right sides
    for i in range(2):
        ts, side = _compute_local_kinematics(data_windowed, data_side, i)

        # Compute dynamic amplitude threshold based on the last 3 cycles
        if len(kinematics_data["cycles"][side]) >= 3:
            min_amplitude_threshold = np.median(
                [c["range"] for c in kinematics_data["cycles"][side][-3:]]
            )
        else:
            min_amplitude_threshold = None

        # Compute mean position threshold over the current window duration
        ts_full = kinematics_data["ts_full"][side]
        if (
            ts_full is not None
            and len(ts_full.time) > 0
            and len(kinematics_data["cycles"][side]) >= 3
        ):
            ts_duration = ts_full.time[-1] - ts_full.time[0]

            if ts_duration > LIMIT_DURATION_CURRENT_WINDOW:
                key_data = next(iter(ts_full.data))
                ts_cropped = ts_full.get_ts_after_time(
                    ts_full.time[-1] - LIMIT_DURATION_CURRENT_WINDOW
                )
                mean_position_threshold = float(
                    ts_cropped.data[key_data][:, 0].mean()
                )
            else:
                mean_position_threshold = None
        else:
            mean_position_threshold = None

        # Detect and segment propulsion cycles using thresholds
        cycles_metrics = propulsion_patterns.detect_propulsion_cycles(
            ts,
            min_amplitude_threshold=min_amplitude_threshold,
            mean_position_threshold=mean_position_threshold,
        )
        cycles_ts_list = propulsion_patterns.segment_propulsion_cycles(
            ts, cycles_metrics
        )

        # Analyze and classify each validated cycle into one of the four common
        # push patterns (PM, SC, SLOP and DLOP)
        cycles_analyzed = []

        for ts_cycle in cycles_ts_list:
            cycle_analyzed = propulsion_patterns.analyse_propulsion_cycle(
                ts_cycle
            )
            cycles_analyzed.append(cycle_analyzed)

        current_window_data[side]["ts"] = ts
        current_window_data[side]["cycles"] = cycles_analyzed

    return current_window_data


def _update_data_cycles(
    cycles: dict[Literal["left", "right"], list[PushCycle]],
    current_window_data: dict[Literal["left", "right"], KtkDataAndCycles],
) -> dict[Literal["left", "right"], list[PushCycle]]:
    """
    Update global cycle history upon cycle detection.

    Appends newly identified propulsion cycles to the continuous historical log
    """
    try:
        sides: tuple[Literal["left", "right"], Literal["left", "right"]] = (
            "left",
            "right",
        )
        for side in sides:
            # Skip if no cycles were detected for this side in the
            # current window
            current_cycles = current_window_data[side]["cycles"]
            if current_cycles is None or not current_cycles:
                continue

            last_cycle = current_cycles[-1]

            # If history cycles is empty for this side, safely append the
            # first cycle
            if len(cycles[side]) == 0:
                cycles[side].append(last_cycle)

            else:
                in_push_time = float(last_cycle["in_push"]["time"])
                end_push_time = float(cycles[side][-1]["end_push"]["time"])

                # Check if a new cycle started after the previous one ended
                if in_push_time > end_push_time:
                    cycles[side].append(last_cycle)

    except Exception as e:
        print(f"_update_data_cycles : {e}")

    return cycles


def _update_ts_full(
    ts_full: dict[Literal["left", "right"], ktk.TimeSeries | None],
    current_window_data: dict[Literal["left", "right"], KtkDataAndCycles],
) -> dict[Literal["left", "right"], ktk.TimeSeries | None]:
    """Update the global timeserie of Meta2 with newly detected timeserie."""
    try:
        sides: tuple[Literal["left", "right"], Literal["left", "right"]] = (
            "left",
            "right",
        )
        for side in sides:
            ts = current_window_data[side]["ts"]

            if ts is None:
                continue

            current_ts = ts_full[side]

            # If history timeserie is empty for this side, safely get the
            # first timeserie
            if current_ts is None:
                ts_full[side] = ts
            else:
                # Cut the timeserie to merge after the previous one ended
                ts_to_merge = ts.get_ts_after_time(
                    current_ts.time[-1], inclusive=False
                )

                current_ts.time = np.concatenate(
                    [current_ts.time, ts_to_merge.time]
                )

                for key in current_ts.data:
                    current_ts.data[key] = np.concatenate(
                        [current_ts.data[key], ts_to_merge.data[key]],
                        axis=0,
                    )
                ts_full[side] = current_ts

    except Exception as e:
        print(f"_update_ts_full : {e}")

    return ts_full


def _send_data_godot(
    new_cycle_send: dict[Literal["left", "right"], int],
    cycles: dict[Literal["left", "right"], list[PushCycle]],
) -> dict[Literal["left", "right"], int]:
    """
    Send computed kinematics metrics to Godot upon cycle detection.

    Streams the median push frequency and push pattern geometry of the last
    three cycles via python_bridge, then increments the sent cycle counter.
    """
    sides: tuple[Literal["left", "right"], Literal["left", "right"]] = (
        "left",
        "right",
    )

    for side in sides:
        if (
            len(cycles[side]) >= 3
            and len(cycles[side]) == new_cycle_send[side]
        ):
            mean_push_frequency = float(
                np.median(
                    [
                        cycles[side][-1]["push_frequency"],
                        cycles[side][-2]["push_frequency"],
                        cycles[side][-3]["push_frequency"],
                    ]
                )
            )

            last_push_pattern_1 = cycles[side][-1][
                "normalised_push_pattern"
            ].tolist()
            last_push_pattern_2 = cycles[side][-2][
                "normalised_push_pattern"
            ].tolist()
            last_push_pattern_3 = cycles[side][-3][
                "normalised_push_pattern"
            ].tolist()

            label_push_pattern = str(cycles[side][-1]["label_push_pattern"])

            data = {
                side: {
                    "mean_push_frequency": mean_push_frequency,
                    "last_push_pattern_1": last_push_pattern_1,
                    "last_push_pattern_2": last_push_pattern_2,
                    "last_push_pattern_3": last_push_pattern_3,
                    "label_push_pattern": label_push_pattern,
                }
            }

            sender.send({"command": "biofeedback_update", "data": data})

            new_cycle_send[side] += 1

    return new_cycle_send


def _initialize_data_side(arg: Arg) -> list[DataSide]:
    """Initialize and structures calibration coordinates for both sides."""
    # Get and convert coordinates to homogeneous arrays [X, Y, Z, 1.0]
    coordinates_left_wheel_center = np.array(
        [list(arg["coordinates_left_wheel_center"]) + [1.0]],
    )

    coordinates_right_wheel_center = np.array(
        [list(arg["coordinates_right_wheel_center"]) + [1.0]],
    )

    coordinates_left_hand = np.array(
        [list(arg["coordinates_left_hand"]) + [1.0]],
    )
    coordinates_right_hand = np.array(
        [list(arg["coordinates_right_hand"]) + [1.0]],
    )

    # Set a dictionnary of side-specific metadata and tracking IDs
    data_side: list[DataSide] = [
        {
            "id_streaming": str(ID_LEFT_HAND_CLUSTER),
            "local_meta2": coordinates_left_hand,
            "side": "left",
            "wheel_center": coordinates_left_wheel_center,
        },
        {
            "id_streaming": str(ID_RIGHT_HAND_CLUSTER),
            "local_meta2": coordinates_right_hand,
            "side": "right",
            "wheel_center": coordinates_right_wheel_center,
        },
    ]

    return data_side


def _get_windowed_data(
    data: dict[str, ktk.TimeSeries],
    limit_duration: float,
) -> dict[str, ktk.TimeSeries]:
    """
    Slice the latest N seconds of the time series.

    This windowing approach is used to optimize real-time processing.
    """
    data_windowed = {}

    if limit_duration == 0:
        return data

    # Iterate through rigid bodies :
    # simulator frame (ID_WHEELCHAIR_SIMULATOR),
    # left forearm (ID_LEFT_HAND_CLUSTER),
    # right forearm (ID_RIGHT_HAND_CLUSTER)
    for key in [
        ID_WHEELCHAIR_SIMULATOR,
        ID_LEFT_HAND_CLUSTER,
        ID_RIGHT_HAND_CLUSTER,
    ]:
        t_end = data[key].time[-1]
        t_start = max(data[key].time[0], t_end - limit_duration)

        data_windowed[key] = data[key].get_ts_between_times(
            t_start, t_end, inclusive=True
        )

    return data_windowed


def _compute_local_kinematics(
    data_windowed: dict[str, ktk.TimeSeries],
    data_side: list[DataSide],
    n: int,
) -> tuple[ktk.TimeSeries, Literal["left", "right"]]:
    """
    Transform tracking data into local kinematics.

    Filters and processes timeseries data for a single wheelchair side.
    """
    # Extract side-specific configuration and streaming tracking ID
    id_streaming = data_side[n]["id_streaming"]
    side = data_side[n]["side"]

    # Estimate second metacarpal (Meta2) position using the forearm
    # cluster reference frame
    ts = ktk.TimeSeries()
    ts.time = data_windowed[id_streaming].time

    ts.data[f"Meta2{side}"] = ktk.geometry.matmul(
        data_windowed[id_streaming].data[id_streaming],
        data_side[n]["local_meta2"],
    )

    # Find the common overlapping time window and resample the forearm
    # signals onto the simulator frame's timeline
    t_min = max(ts.time[0], data_windowed[ID_WHEELCHAIR_SIMULATOR].time[0])
    t_max = min(ts.time[-1], data_windowed[ID_WHEELCHAIR_SIMULATOR].time[-1])

    ts_data = data_windowed[ID_WHEELCHAIR_SIMULATOR].get_ts_between_times(
        t_min, t_max
    )
    ts = ts.get_ts_between_times(t_min, t_max)

    ts = ts.resample(ts_data.time)

    # Transform Meta2 coordinates from the global tracking system to the
    # simulator's local coordinate system
    ts.data[f"Meta2{side}"] = ktk.geometry.get_local_coordinates(
        global_coordinates=ts.data[f"Meta2{side}"],
        reference_frames=ts_data.data[ID_WHEELCHAIR_SIMULATOR],
    )

    # Center coordinates relative to the calibrated wheel center
    wheel_center_local = data_side[n]["wheel_center"][0, :3]
    ts.data[f"Meta2{side}"][:, 0] -= wheel_center_local[0]
    ts.data[f"Meta2{side}"][:, 1] -= wheel_center_local[1]
    ts.data[f"Meta2{side}"][:, 2] -= wheel_center_local[2]

    # Set sample rate constant
    dt = np.median(np.diff(ts.time))
    time_uniform = np.arange(ts.time[0], ts.time[-1], dt)
    ts = ts.resample(time_uniform)

    # Filter butterworth order 4 with cut frequency of 6Hz
    ts = ktk.filters.butter(ts, fc=6, order=4)

    # Add velocity and acceleration timeseries
    ts_df = ktk.filters.deriv(ts, n=1)
    ts_dff = ktk.filters.deriv(ts, n=2)

    ts = ts.get_ts_before_index(len(ts.time) - 1)
    ts.data[f"Meta2{side}_df"] = ts_df.data[f"Meta2{side}"][:, 0]
    ts = ts.get_ts_before_index(len(ts.time) - 1)
    ts.data[f"Meta2{side}_dff"] = ts_dff.data[f"Meta2{side}"][:, 0]

    return ts, side


def _init_variables() -> None:
    """Initialize runtime and kinematics state variables."""
    _runtime_state["run_mode"] = "stop"
    _runtime_state["data"] = None
    _runtime_state["current_window_data"] = None
    _runtime_state["new_cycle_log"] = {"left": 1, "right": 1}
    _runtime_state["new_cycle_send"] = {"left": 3, "right": 3}

    kinematics_data["cycles"] = {"left": [], "right": []}
    kinematics_data["ts_full"] = {"left": None, "right": None}


def _execute_analysis_pipeline(
    _runtime_state: RuntimeState,
    kinematics_data: KinematicsData,
    arg: Arg,
    LIMIT_DURATION_CURRENT_WINDOW: float,
) -> tuple[float, float]:
    """Run kinematic analysis and distribute results."""
    if _runtime_state["data"] is None:
        return

    _runtime_state["current_window_data"] = _analyze_current_window(
        _runtime_state["data"],
        arg,
        kinematics_data["cycles"],
        limit_duration=LIMIT_DURATION_CURRENT_WINDOW,
    )

    if _runtime_state["current_window_data"] is None:
        return

    kinematics_data["cycles"] = _update_data_cycles(
        kinematics_data["cycles"],
        _runtime_state["current_window_data"],
    )
    kinematics_data["ts_full"] = _update_ts_full(
        kinematics_data["ts_full"],
        _runtime_state["current_window_data"],
    )

    _runtime_state["new_cycle_send"] = _send_data_godot(
        _runtime_state["new_cycle_send"],
        kinematics_data["cycles"],
    )

    return


# %% Main
if __name__ == "__main__":
    """
    Execution script for testing the script in standalone. Normally the
    biofeedback functions are called by Godot using the python bridge.

    This script is directly functional out-of-the-box. To run it, ensure you:
    1. Start the live data streaming from Motive (OptiTrack).
    2. Configure the physical properties in the 'arg' dictionary.
    3. Wear both tracking clusters configured in Motive with the following IDs:
        - ID Reference Wheelchair simulator (ID_WHEELCHAIR_SIMULATOR)
        - ID Left Hand Cluster (ID_LEFT_HAND_CLUSTER)
        - ID Right Hand Cluster (ID_RIGHT_HAND_CLUSTER)

    Press Ctrl+C in the terminal to safely stop the stream and close the
    biofeedback.
    """

    arg: Arg = {
        "coordinates_left_wheel_center": (
            -0.150,
            0.300,
            -0.750,
        ),
        "coordinates_right_wheel_center": (
            -0.145,
            0.300,
            -0.200,
        ),
        "coordinates_left_hand": (-0.145, 0.050, 0.020),
        "coordinates_right_hand": (0.020, -0.160, 0.000),
        "wheel_diameter": 0.54,
    }

    try:
        while True:
            biofeedback_update(**arg)
    except KeyboardInterrupt:
        print("Biofeedback closed")
        biofeedback_stop()
