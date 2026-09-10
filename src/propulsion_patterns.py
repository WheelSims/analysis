"""
Detect, segment, and classify propulsion patterns based on kinematics.

Coordinate System & Reference Frame Conventions
------------------------------------------------
All kinematic analysis and geometric classification algorithms in this module
assume that the hand trajectory data is expressed in a local, wheel-centered
reference frame:
- Origin (0, 0): Coincides with the center of the wheelchair wheel.
- X-axis: Anteroposterior direction (positive pointing forward).
- Y-axis: Vertical direction (positive pointing upward).
- Z-axis: Mediolateral direction.

If your raw tracking data is in a global reference frame, you must translate
and rotate the coordinates to align with this local sagittal plane convention
before passing them to any functions in this module.
"""

from typing import TypedDict

import kineticstoolkit as ktk
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

# %% Public variables
# Area ratio thresholds used to discriminate between Semi-Circular (SC),
# Single-Loop (SLOP), and Double-Loop (DLOP) propulsion techniques.
PATTERN_A2_SC_THRESHOLD = 0.75
PATTERN_A2_SLOP_THRESHOLD = 0.75

# Maximum allowed geometric deviation for pattern validation
MAX_DEVIATION_THRESHOLD = 0.1

# Minimum required duration for a valid propulsion cycle (in seconds)
MIN_CYCLE_DURATION = 0.4

# Minimum peak velocity required to filter out parasitic movements (m/s)
MIN_PEAK_VELOCITY = 0.2

# Standard 24" wheelchair wheel diameter (m)
WHEEL_DIAMETER = 0.54

# Time window used to compute the reference mean hand position (s)
MEAN_POSITION_WINDOW_DURATION = 3.0


# %% Typing classes
class SignedAreas(TypedDict):
    """
    Signed geometric areas and trajectories between propulsion phases.

    sign :
        Sign of the deviation between recovery and push curves.
    area :
        Calculated surface area between the segmented trajectories.
    recovery_phase :
        Array of coordinates representing the recovery path for this segment.
    push_phase :
        Array of coordinates representing the push path for this segment.
    """

    sign: str  # "positive" or "negative"
    area: float
    recovery_phase: np.ndarray
    push_phase: np.ndarray


class SegmentedCycle(TypedDict):
    """
    Temporal boundaries and key positions for a segmented propulsion cycle.

    in_push :
        Time (s) and coordinate value (m) marking the start of the push phase.
    end_push :
        Time (s) and coordinate value (m) marking the end of the push phase.
    """

    in_push: dict[str, float]  # where str is "time" or "value"
    end_push: dict[str, float]  # where str is "time" or "value"


class FilteredCycle(TypedDict):
    """
    Calculated metrics for filtering a single full propulsion cycle.

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
    """

    in_push: dict[str, float]  # where str is "time" or "value"
    recovery: dict[str, float]  # where str is "time" or "value"
    end_push: dict[str, float]  # where str is "time" or "value"
    range: float
    velocity_max: float


class AnalyzedCycle(TypedDict):
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

    in_push: dict[str, float]  # where str is "time" or "value"
    recovery: dict[str, float]  # where str is "time" or "value"
    end_push: dict[str, float]  # where str is "time" or "value"
    range: float
    velocity_max: float
    push_frequency: float
    normalised_push_pattern: np.ndarray
    areas: list[SignedAreas] | None
    A1: float | None
    A2: float | None
    label_push_pattern: str | None


# %% Public functions
def detect_propulsion_cycles(
    ts_propulsion_cycles: ktk.TimeSeries,
    min_cycle_duration: float = MIN_CYCLE_DURATION,
    min_peak_velocity: float = MIN_PEAK_VELOCITY,
    min_amplitude_threshold: float | None = None,
    mean_position_threshold: float | None = None,
) -> list[SegmentedCycle]:
    """
    Detect and segment propulsion cycles from hand kinematic data.

    The function identifies propulsion cycles from a kinematic TimeSeries
    recorded in the sagittal plane and segments the signal into individual
    propulsion cycles.

    Notes
    -----
    The input TimeSeries coordinates must be expressed in a wheel-centered
    local coordinate system (origin at the wheel center, X-axis positive
    forward, Y-axis positive upward). Please refer to the module documentation
    for more details.
    """
    ts_calc = ts_propulsion_cycles.copy()

    # Set sample rate constant if necessary
    dt = np.diff(ts_calc.time)

    if not np.allclose(dt, np.median(dt), rtol=1e-3, atol=1e-6):
        dt_median = np.median(dt)

        time_uniform = np.arange(
            ts_calc.time[0],
            ts_calc.time[-1],
            dt_median,
        )

        ts_calc = ts_calc.resample(time_uniform)

    # Add velocity and acceleration timeseries
    ts_df = ktk.filters.deriv(ts_calc, n=1)
    ts_dff = ktk.filters.deriv(ts_calc, n=2)

    key_data = next(iter(ts_calc.data))

    ts_calc = ts_calc.get_ts_before_index(len(ts_calc.time) - 1)
    ts_calc.data[f"{key_data}_df"] = ts_df.data[key_data][:, 0]

    ts_calc = ts_calc.get_ts_before_index(len(ts_calc.time) - 1)
    ts_calc.data[f"{key_data}_dff"] = ts_dff.data[key_data][:, 0]

    # Cycle detection upon velocity zero-crossing with temporal criterion
    # (duration > min_cycle_duration s)
    pos_x = ts_calc.data[key_data][:, 0]
    vel_x = ts_calc.data[f"{key_data}_df"]

    if np.all(vel_x >= 0) or np.all(vel_x <= 0):
        return []

    try:
        ts_events = ktk.cycles.detect_cycles(
            ts_calc,
            f"{key_data}_df",
            thresholds=(0.0, 0.0),
            event_names=("push", "recovery"),
        )
    except Exception as e:
        print(e)
        return []

    events = [e for e in ts_events.events if e.name != "_"]

    if len(events) < 3:
        return []

    cycles: list[FilteredCycle] = []

    for i in range(len(events) - 2):
        if (
            events[i].name == "push"
            and events[i + 1].name == "recovery"
            and events[i + 2].name == "push"
        ):
            index_t = ts_calc.get_index_at_time(events[i].time)
            index_t1 = ts_calc.get_index_at_time(events[i + 1].time)
            index_t2 = ts_calc.get_index_at_time(events[i + 2].time)

            t = events[i].time
            t1 = events[i + 1].time
            t2 = events[i + 2].time

            delta_t = events[i + 2].time - events[i].time

            if delta_t > min_cycle_duration:
                ts_cycle = ts_calc.get_ts_between_times(t, t2)
                ts_cycle.time = np.linspace(0, 100, len(ts_cycle.time))

                cycles.append(
                    {
                        "in_push": {
                            "time": float(t),
                            "value": float(pos_x[index_t]),
                        },
                        "recovery": {
                            "time": float(t1),
                            "value": float(pos_x[index_t1]),
                        },
                        "end_push": {
                            "time": float(t2),
                            "value": float(pos_x[index_t2]),
                        },
                        "range": float(pos_x[index_t1] - pos_x[index_t]),
                        "velocity_max": float(
                            np.nanmax(vel_x[index_t:index_t2]),
                        ),
                    },
                )

    # Kinematic criterion #1: minimum amplitude based on the general
    # amplitude (median) of the last 3 cycles
    if min_amplitude_threshold is None:
        cycles = _filter_cycles_by_velocity_and_amplitude(cycles)
    else:
        cycles = _filter_cycles_by_velocity_and_amplitude(
            cycles, min_amplitude_threshold
        )
    # Kinematic criterion #2: condition to cross the mean
    # anterior-posterior position computed over the last 3 seconds
    cycles = _filter_cycles_by_mean_crossing(
        cycles, ts_calc, mean_position_threshold=mean_position_threshold
    )

    # Build the cycle metrics dictionary
    cycles_metrics: list[SegmentedCycle] = [
        {"in_push": cycle["in_push"], "end_push": cycle["end_push"]}
        for cycle in cycles
    ]

    return cycles_metrics


def segment_propulsion_cycles(
    ts_propulsion_cycles: ktk.TimeSeries,
    cycles_metrics: list[SegmentedCycle],
) -> list[ktk.TimeSeries]:
    """
    Split a continuous propulsion TimeSeries into individual cycle TimeSeries.

    Using the validated temporal boundaries (start of push and end of push)
    from the detected cycles, this function extracts and isolates a subset
    TimeSeries for each individual propulsion cycle.
    """
    cycles_ts_list: list[ktk.TimeSeries] = []

    for cycle in cycles_metrics:
        t_start = cycle["in_push"]["time"]
        t_end = cycle["end_push"]["time"]

        ts_single_cycle = ts_propulsion_cycles.get_ts_between_times(
            t_start, t_end
        )

        cycles_ts_list.append(ts_single_cycle)

    return cycles_ts_list


def analyse_propulsion_cycle(
    ts_cycle: ktk.TimeSeries,
    recovery_time: float | None = None,
) -> AnalyzedCycle:
    """
    Analyse and extract kinematic metrics from a single propulsion cycle.

    This function processes an individual cycle's TimeSeries by checking and
    enforcing a uniform sampling rate, computing local velocities and
    accelerations via differentiation, detecting key phases (push and recovery
    ), normalizing the pattern trajectory, and classifying the propulsion
    technique (PM, SC, SLOP, or DLOP) using geometric criteria.

    Notes
    -----
    The input TimeSeries coordinates must be expressed in a wheel-centered
    local coordinate system (origin at the wheel center, X-axis positive
    forward, Y-axis positive upward). Please refer to the module documentation
    for more details.
    """
    ts_calc = ts_cycle.copy()

    # Set sample rate constant if necessary
    dt = np.diff(ts_calc.time)

    if not np.allclose(dt, np.median(dt), rtol=1e-3, atol=1e-6):
        dt_median = np.median(dt)

        time_uniform = np.arange(
            ts_calc.time[0],
            ts_calc.time[-1],
            dt_median,
        )

        ts_calc = ts_calc.resample(time_uniform)

    # Add velocity and acceleration timeseries
    ts_df = ktk.filters.deriv(ts_calc, n=1)
    ts_dff = ktk.filters.deriv(ts_calc, n=2)

    key_data = next(iter(ts_calc.data))

    ts_calc = ts_calc.get_ts_before_index(len(ts_calc.time) - 1)
    ts_calc.data[f"{key_data}_df"] = ts_df.data[key_data][:, 0]
    ts_calc = ts_calc.get_ts_before_index(len(ts_calc.time) - 1)
    ts_calc.data[f"{key_data}_dff"] = ts_dff.data[key_data][:, 0]

    delta_t = ts_calc.time[-1] - ts_calc.time[0]

    ts_calc = ts_calc.add_event(ts_calc.time[0], "in_push")
    ts_calc = ts_calc.add_event(ts_calc.time[-1], "end_push")

    ts_normalised = ktk.cycles.time_normalize(
        ts_calc, "in_push", "end_push", n_points=101
    )
    normalised_push_pattern = ts_normalised.data[key_data][:, 0:3]

    vel_x = ts_calc.data[f"{key_data}_df"]

    index_recovery = np.nanargmax(ts_calc.data[key_data][:, 0])

    # Classify each validated cycle into one of the four common
    # push patterns (PM, SC, SLOP and DLOP)
    label_push_pattern, A1, A2, signed_areas = classify_push_pattern(
        ts_calc, recovery_time=recovery_time
    )

    # Build the analyzed cycle metrics dictionary
    cycle_analyzed: AnalyzedCycle = {
        "in_push": {
            "time": float(ts_calc.time[0]),
            "value": float(ts_calc.data[key_data][0, 0]),
        },
        "recovery": {
            "time": float(ts_calc.time[index_recovery]),
            "value": float(ts_calc.data[key_data][index_recovery, 0]),
        },
        "end_push": {
            "time": float(ts_calc.time[-1]),
            "value": float(ts_calc.data[key_data][-1, 0]),
        },
        "range": float(
            ts_calc.data[key_data][index_recovery, 0]
            - ts_calc.data[key_data][0, 0]
        ),
        "velocity_max": float(
            np.nanmax(vel_x),
        ),
        "push_frequency": float(1 / delta_t),
        "normalised_push_pattern": normalised_push_pattern,
        "areas": signed_areas,
        "A1": A1,
        "A2": A2,
        "label_push_pattern": label_push_pattern,
    }

    return cycle_analyzed


def extract_propulsion_phases(
    ts_propulsion_cycles: ktk.TimeSeries,
    in_push_time: float | None = None,
    end_push_time: float | None = None,
    recovery_time: float | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Split a propulsion cycle into the push and recovery phases.

    By default, if "in_push_time" and "end_push_time" are not provided, they
    default to the start and end times of the input TimeSeries, respectively.
    However, if one is specified, the other must also be specified.

    If "recovery_time" is not specified, it is calculated automatically by
    identifying the point of change in movement direction (maximum coordinate
    along the anteroposterior axis) within the defined time range.
    """
    # Force both boundaries to be provided together, or not at all
    if (in_push_time is None) != (end_push_time is None):
        raise ValueError(
            "Both 'in_push_time' and 'end_push_time' must be provided "
            "together, or both must be None to use the TimeSeries limits."
        )

    # Use the start and end of the TimeSeries as default boundaries
    if in_push_time is None:
        in_push_time = float(ts_propulsion_cycles.time[0])
    if end_push_time is None:
        end_push_time = float(ts_propulsion_cycles.time[-1])

    key_data = next(iter(ts_propulsion_cycles.data))

    # Calculate recovery transition if not provided
    if recovery_time is None:
        ts_sub = ts_propulsion_cycles.get_ts_between_times(
            in_push_time, end_push_time
        )
        idx_max = np.nanargmax(ts_sub.data[key_data][:, 0])
        recovery_time = float(ts_sub.time[idx_max])

    # Final extraction of the phases
    idx_in_push = ts_propulsion_cycles.get_index_at_time(in_push_time)
    idx_recovery = ts_propulsion_cycles.get_index_at_time(recovery_time)
    idx_end_push = ts_propulsion_cycles.get_index_at_time(end_push_time)

    if idx_recovery > idx_in_push:
        push_phase = ts_propulsion_cycles.get_ts_between_indexes(
            idx_in_push, idx_recovery, inclusive=True
        ).data[key_data][:, 0:2]
    else:
        push_phase = None

    if idx_end_push > idx_recovery:
        recovery_phase = ts_propulsion_cycles.get_ts_between_indexes(
            idx_recovery, idx_end_push, inclusive=True
        ).data[key_data][:, 0:2]
    else:
        recovery_phase = None

    return push_phase, recovery_phase


def classify_push_pattern(
    ts_cycle: ktk.TimeSeries,
    in_push_time: float | None = None,
    end_push_time: float | None = None,
    recovery_time: float | None = None,
    *,
    max_deviation_threshold: float = MAX_DEVIATION_THRESHOLD,
    pattern_a2_sc_threshold: float = PATTERN_A2_SC_THRESHOLD,
    pattern_a2_slop_threshold: float = PATTERN_A2_SLOP_THRESHOLD,
) -> tuple[
    str,
    float | None,
    float | None,
    list[SignedAreas] | None,
]:
    """
    Classify the wheelchair propulsion pattern of a single propulsion cycle.

    The classification is based on the geometric criteria A1 and A2 computed
    from the hand trajectory during the recovery phase.

    By default, if "in_push_time" and "end_push_time" are not provided, they
    default to the start and end times of the input TimeSeries, respectively.
    However, if one is specified, the other must also be specified.

    If "recovery_time" is not specified, it is calculated automatically by
    identifying the point of change in movement direction (maximum coordinate
    along the anteroposterior axis) within the defined time range.

    The decision rules are:

    - A1 < 1: Pumping (PM)
    - A2 <= -PATTERN_A2_SC_THRESHOLD: Semi-Circular (SC)
    - A2 >= PATTERN_A2_SLOP_THRESHOLD: Single-Loop (SLOP)
    - Otherwise: Double-Loop (DLOP)

    Notes
    -----
    The input TimeSeries coordinates must be expressed in a wheel-centered
    local coordinate system (origin at the wheel center, X-axis positive
    forward, Y-axis positive upward). Please refer to the module documentation
    for more details.

    If these conditions are not satisfied, the computed geometric criteria
    (A1 and A2) may be incorrect, resulting in an incorrect propulsion
    pattern classification.
    """
    # Force both boundaries to be provided together, or not at all
    if (in_push_time is None) != (end_push_time is None):
        raise ValueError(
            "Both 'in_push_time' and 'end_push_time' must be provided "
            "together, or both must be None to use the TimeSeries limits."
        )

    # Extract push and recovery phase
    push_phase, recovery_phase = extract_propulsion_phases(
        ts_cycle,
        in_push_time=in_push_time,
        end_push_time=end_push_time,
        recovery_time=recovery_time,
    )

    if push_phase is None or recovery_phase is None:
        A1 = None
        A2 = None
        signed_areas = None

    else:
        signed_areas = _compute_geometric_zones(push_phase, recovery_phase)

        A1 = _compute_a1_score(
            recovery_phase, push_phase, max_deviation_threshold
        )
        A2 = _compute_a2_score(signed_areas)

        if A1 < 1:
            label_push_pattern = "Pumping (PM)"
        elif A2 <= -pattern_a2_sc_threshold:
            label_push_pattern = "Semi-Circular (SC)"
        elif A2 >= pattern_a2_slop_threshold:
            label_push_pattern = "Single-Loop (SLOP)"
        # A2 < pattern_a2_slop_threshold and A2 > -pattern_a2_sc_threshold
        else:
            label_push_pattern = "Double-Loop (DLOP)"

    return label_push_pattern, A1, A2, signed_areas


def plot_bilateral_cycles(
    dict_ts_propulsion_cycles: dict[str, ktk.TimeSeries],
    dict_cycles: (
        dict[str, list[AnalyzedCycle]] | dict[str, list[SegmentedCycle]]
    ),  # where str is "left" or "right"
) -> None:
    """
    Plot bilateral hand kinematics with highlighted propulsion cycles.

    This function generates a single Matplotlib figure with two vertically
    stacked subplots displaying the continuous anteroposterior position
    of the hand over time for both the left (top) and right (bottom) sides.
    Individual detected propulsion cycles are visually highlighted in the
    background using alternating shaded vertical spans (axvspan).
    """
    plt.figure()
    plt.suptitle("Bilateral kinematics")

    for side in ("left", "right"):
        ts_full = dict_ts_propulsion_cycles[side]

        if ts_full is None:
            continue

        key_data = next(iter(ts_full.data))

        if side == "left":
            plt.subplot(2, 1, 1)
        else:
            plt.subplot(2, 1, 2)

        plt.title("Position")
        colors = [(1, 0, 0), (0.5, 0.25, 0.25)]

        for i, cycle in enumerate(dict_cycles[side]):
            start = cycle["in_push"]["time"]
            end = cycle["end_push"]["time"]
            color = colors[i % 2]

            plt.axvspan(float(start), float(end), color=color, alpha=0.3)

        plt.plot(
            ts_full.time,
            ts_full.data[key_data][:, 0],
            label=key_data,
        )
        plt.xlabel("Time (s)")
        plt.legend()

        plt.tight_layout()


def plot_unilateral_push_patterns(
    ts_propulsion_cycles: ktk.TimeSeries,
    cycles: list[AnalyzedCycle],
    side: str = "unspecified",  # or "left" or "right"
    wheel_diameter: float = WHEEL_DIAMETER,
) -> None:
    """
    Plot unilateral propulsion patterns.

    Generates a multipage grid of subplots (up to 12 subplots per page,
    arranged in 2 rows by 6 columns). For each cycle, it renders:
    - The hand's 2D trajectory in the sagittal plane (push phase as a solid
      line, recovery phase as a dashed line).
    - The wheel's outer boundary (dotted circle) centered at (0, 0).
    - The positive (red) and negative (green) geometric classification zones.
    - The classification metrics (A1, A2) and the assigned pattern label.

    Notes
    -----
    The input TimeSeries coordinates must be expressed in a wheel-centered
    local coordinate system (origin at the wheel center, X-axis positive
    forward, Y-axis positive upward). Please refer to the module documentation
    for more details.
    """
    key_data = next(iter(ts_propulsion_cycles.data))
    num_cycles = len(cycles)

    if num_cycles == 0:
        print(f"No cycles detected to plot for {side} side.")
        return

    n_cols = 6
    n_rows = 2
    max_cycles_per_page = n_cols * n_rows

    total_pages = int(np.ceil(num_cycles / max_cycles_per_page))

    for page in range(1, total_pages + 1):
        start_idx = (page - 1) * max_cycles_per_page
        end_idx = min(start_idx + max_cycles_per_page, num_cycles)
        page_cycles = cycles[start_idx:end_idx]

        fig = plt.figure()

        plt.suptitle(
            f"\
                push pattern {side} side | Page {page}/{total_pages}\
                | ({num_cycles} cycles total)",
            fontsize=14,
            weight="bold",
            y=0.98,
        )

        for i, cycle in enumerate(page_cycles, start=1):
            ax = plt.subplot(n_rows, n_cols, i)

            # Draw positive and negative areas
            if cycle["areas"] is not None:
                zones = cycle["areas"]
                for zone in zones:
                    _points = np.vstack(
                        (zone["recovery_phase"], zone["push_phase"][::-1]),
                    )
                    _facecolor = (
                        "green" if zone["sign"] == "negative" else "red"
                    )
                    _label = (
                        "negative area"
                        if zone["sign"] == "negative"
                        else "positive area"
                    )

                    poly_param = MplPolygon(
                        _points,
                        closed=True,
                        fill=True,
                        facecolor=_facecolor,
                        alpha=0.4,
                        label=_label,
                    )
                    ax.add_patch(poly_param)

            # Split the time-series cycle into recovery and push phases

            if ts_propulsion_cycles is None:
                continue

            recovery_phase = ts_propulsion_cycles.get_ts_between_times(
                cycle["recovery"]["time"],
                cycle["end_push"]["time"],
            ).data[key_data][:, 0:2]
            push_phase = ts_propulsion_cycles.get_ts_between_times(
                cycle["in_push"]["time"],
                cycle["recovery"]["time"],
            ).data[key_data][:, 0:2]

            # Draw the push phase
            ax.plot(
                push_phase[0, 0],
                push_phase[0, 1],
                color="black",
                marker="o",
                markersize=4,
                linewidth=2,
                zorder=4,
                label="start push phase",
            )
            ax.plot(
                push_phase[:, 0],
                push_phase[:, 1],
                color="black",
                linewidth=1,
                zorder=4,
                label="push phase",
            )

            # # Draw the recovery phase
            ax.plot(
                recovery_phase[:, 0],
                recovery_phase[:, 1],
                color="black",
                linewidth=1,
                zorder=4,
                label="recovery phase",
                linestyle="--",
            )

            # Draw the wheel
            circle = plt.Circle(
                (
                    0,
                    0,
                ),
                wheel_diameter / 2,
                fill=False,
                linestyle="dotted",
                label="wheel",
            )
            ax.add_patch(circle)

            ax.set_xlim(-0.4, 0.6)
            ax.set_ylim(-0.4, 0.6)
            ax.set_aspect("equal")
            global_cycle_number = start_idx + i
            A1 = cycle["A1"]
            A2 = cycle["A2"]
            ax.set_title(
                f"\
                    Push n°{global_cycle_number} \n \
                    A1 : {A1:.2f}   ¦   A2 : {A2:.2f} \n \
                    {cycle['label_push_pattern']}",
            )

        # Create a global legend for all subplots
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            dict(zip(labels, handles, strict=True)).values(),
            dict(zip(labels, handles, strict=True)).keys(),
        )

        # Adjust subplot spacing
        fig.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)


# %% Private functions
def _compute_geometric_zones(
    push_phase: np.ndarray,
    recovery_phase: np.ndarray,
) -> list[SignedAreas]:
    """
    Compute signed areas between recovery and push trajectories.

    This is achieved by segmenting the signal at curve crossings to
    determine the geometric zones.

    Assumes coordinates are already formatted and centered according to the
    module's reference frame convention.
    """
    recovery = np.array(recovery_phase)
    push = np.array(push_phase)

    # Sort push curve by anteroposterior for interpolation
    push_sorted = push[np.argsort(push[:, 0])]

    y_push_interpolated = np.interp(
        recovery[:, 0],
        push_sorted[:, 0],
        push_sorted[:, 1],
    )

    # Mask to detect if the hand in the recovery crosses the push line
    above = recovery[:, 1] >= y_push_interpolated

    # Ensure last segment is closed
    extended_mask = np.append(above, not above[-1])

    signed_areas: list[SignedAreas] = []
    start_idx = 0
    current_sign = above[0]

    for i in range(1, len(extended_mask)):
        if extended_mask[i] != current_sign:
            current_recovery_phase = recovery[start_idx : i + 1]
            current_push_phase = np.column_stack(
                (
                    recovery[start_idx : i + 1, 0],
                    y_push_interpolated[start_idx : i + 1],
                ),
            )

            # Calculating geometric area using the trapezoidal rule
            dx = current_recovery_phase[1:, 0] - current_recovery_phase[:-1, 0]
            mean_recovery_y = (
                current_recovery_phase[1:, 1] + current_recovery_phase[:-1, 1]
            ) / 2.0
            mean_push_y = (
                current_push_phase[1:, 1] + current_push_phase[:-1, 1]
            ) / 2.0

            area = np.sum((mean_recovery_y - mean_push_y) * dx)

            signed_areas.append(
                {
                    "sign": ("positive" if current_sign else "negative"),
                    "area": abs(area),
                    "recovery_phase": current_recovery_phase,
                    "push_phase": current_push_phase,
                },
            )

            start_idx = i
            current_sign = extended_mask[i]

    return signed_areas


def _compute_a1_score(
    recovery_phase: np.ndarray,
    push_phase: np.ndarray,
    deviation_max: float = MAX_DEVIATION_THRESHOLD,
) -> float:
    """
    Compute the normalized recovery-phase deviation index.

    The index is calculated relative to the push-phase radius.
    A1 compares the hand deviation during recovery to a reference
    threshold (d_max). Values > 1 indicate large deviation.

    Assumes coordinates are already formatted and centered according to the
    module's reference frame convention.
    """
    # Calculate radial distance of hand positions during push phase
    push_distances = np.sqrt(
        (push_phase[:, 0]) ** 2 + (push_phase[:, 1]) ** 2,
    )

    # Calculate radial distance of hand positions during recovery phase
    distance_hand_wheel_center = np.sqrt(
        (recovery_phase[:, 0]) ** 2 + (recovery_phase[:, 1]) ** 2,
    )

    # Compute absolute deviation between recovery distances and minimum push
    # distance
    min_push_distance = np.min(push_distances)
    deviation = np.sort(
        np.abs(distance_hand_wheel_center - min_push_distance),
    )

    # Median of the upper quartile (75–100%)
    mean_distance_deviation = np.median(
        deviation[int(len(deviation) * 0.75) :],
    )

    A1 = mean_distance_deviation / deviation_max

    return A1


def _compute_a2_score(signed_areas: list[SignedAreas]) -> float:
    """
    Symmetry index based on signed areas.

    A2 = (positive areas - negative areas) / total areas
    Range: [-1, 1]
        +1 --> positive dominance
        -1 --> negative dominance
    """
    Ap = 0.0
    An = 0.0

    # Sum total positive and negative geometric areas
    for area in signed_areas:
        if area["sign"] == "positive":
            Ap += area["area"]
        if area["sign"] == "negative":
            An += area["area"]

    A2 = (Ap - An) / (Ap + An)

    return A2


def _filter_cycles_by_velocity_and_amplitude(
    cycles: list[FilteredCycle],
    min_amplitude_threshold: float | None = None,
    min_peak_velocity: float = MIN_PEAK_VELOCITY,
) -> list[FilteredCycle]:
    """
    Validate propulsion cycles based on kinematic amplitude and peak velocity.

    All cycles must satisfy a minimum peak velocity criterion.

    The amplitude criterion is applied as follows:

    - If at least three valid cycles have already been retained, the reference
      amplitude is computed as the median amplitude "range" of the three
      most recently retained cycles.
    - Otherwise, if "min_amplitude_threshold" is provided, it is used as the
      reference amplitude.
    - If neither condition is met, no amplitude filtering is applied.

    A cycle is retained only if its amplitude is at least 30% of the selected
    reference amplitude.
    """
    cycles_filtered_1: list[FilteredCycle] = []

    for cycle in cycles:
        if cycle["velocity_max"] <= min_peak_velocity:
            continue

        amplitude_reference = None

        if len(cycles_filtered_1) >= 3:
            amplitude_reference = float(
                np.median(
                    [
                        cycles_filtered_1[-1]["range"],
                        cycles_filtered_1[-2]["range"],
                        cycles_filtered_1[-3]["range"],
                    ]
                )
            )
        elif min_amplitude_threshold is not None:
            amplitude_reference = min_amplitude_threshold

        if (
            amplitude_reference is not None
            and cycle["range"] < 0.3 * amplitude_reference
        ):
            continue

        cycles_filtered_1.append(cycle)

    return cycles_filtered_1


def _filter_cycles_by_mean_crossing(
    cycles: list[FilteredCycle],
    ts_propulsion_cycles: ktk.TimeSeries,
    mean_position_threshold: float | None = None,
) -> list[FilteredCycle]:
    """
    Validate propulsion cycles based on mean position crossings.

    Filters cycles to ensure the anterior-posterior position signal crosses the
    reference mean position both upward and downward during the cycle.

    The mean position reference is determined as follows:

    - If "mean_position_threshold" is provided, it is directly used as the
      reference mean position.
    - Otherwise, if the time series duration is at least 3 seconds, the
      reference is computed as the mean position over the last 3 seconds of the
      time series.
    - If neither condition is met, the mean position is computed over the
      entire time series provided.

    Filtering rules:

    - If no "mean_position_threshold" is provided, the first 3 cycles are
      automatically retained without mean crossing checks.
    - For all other cycles (or when a threshold is explicitly passed), a cycle
      is retained only if its anterior-posterior signal crosses the reference
      value in both directions (upward and downward).
    """
    cycles_filtered_2: list[FilteredCycle] = []
    key_data = next(iter(ts_propulsion_cycles.data))
    signal = ts_propulsion_cycles.data[key_data][:, 0]

    if mean_position_threshold is not None:
        mean_value = mean_position_threshold
    else:
        duration_ts = (
            ts_propulsion_cycles.time[-1] - ts_propulsion_cycles.time[0]
        )
        if duration_ts >= MEAN_POSITION_WINDOW_DURATION:
            mean_value = (
                ts_propulsion_cycles.get_ts_after_time(
                    ts_propulsion_cycles.time[-1]
                    - MEAN_POSITION_WINDOW_DURATION
                )
                .data[key_data][:, 0]
                .mean()
            )
        else:
            mean_value = signal.mean()

    for r, cycle in enumerate(cycles):
        # Keep the first three cycles only when using the automatic threshold
        if mean_position_threshold is None and r < 3:
            cycles_filtered_2.append(cycle)
            continue

        t0 = ts_propulsion_cycles.get_index_at_time(cycle["in_push"]["time"])
        t2 = ts_propulsion_cycles.get_index_at_time(cycle["end_push"]["time"])

        segment = signal[t0 : t2 + 1]

        crossed_up = False
        crossed_down = False

        for i in range(len(segment) - 1):
            if segment[i] < mean_value and segment[i + 1] >= mean_value:
                crossed_up = True
            if segment[i] > mean_value and segment[i + 1] <= mean_value:
                crossed_down = True

            if crossed_up and crossed_down:
                break

        if crossed_up and crossed_down:
            cycles_filtered_2.append(cycle)

    return cycles_filtered_2
