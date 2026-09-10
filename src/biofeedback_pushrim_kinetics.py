"""Calculations for pushrim kinetics biofeedback."""

from dataclasses import dataclass
from typing import Any

import kineticstoolkit as ktk
import numpy as np
from kineticstoolkit_extensions import pushrimkinetics as pk
from nextwheel import NextWheel

from nextwheel_dummy import NextWheel as NextWheelDummy

TIME_SPAN = 5  # Number of seconds to show on the rolling plot
N_POINTS = 300  # Number of points in the rolling plot

N_PUSHES = 3  # Number of pushes to calculate on
RESAMPLE_FREQUENCY = 240  # Resample everything to this freq before processing


@dataclass
class PushrimKineticsBiofeedback:
    """Contains the NextWheel instance and processing methods."""

    # The NextWheel instance
    nw: NextWheel | NextWheelDummy = NextWheelDummy()

    # Functions
    def connect(self, ip: str) -> None:
        """
        Connect to the NextWheel.

        Parameters
        ----------
        ip
            IP address of the NextWheel, or "dummy" to simulate a wheel that
            sends prerecorded data.
        """
        if ip == "dummy":
            self.nw = NextWheelDummy()
        else:
            self.nw = NextWheel(ip)
            self.nw.start_streaming()

    def process(self) -> dict:
        """Process data and send back to Godot, called regularly."""
        nextwheel_data = self.nw.fetch()
        return calculate_pushrim_kinetics_biofeedback(nextwheel_data)


# %% Public interface for main.py

# Create the instance
pkb = PushrimKineticsBiofeedback(nw=NextWheel())


def connect(ip: str) -> None:
    """Connect to the NextWheel."""
    pkb.connect(ip)


def process(**kwargs) -> dict:
    """Run the process once."""
    return pkb.process()


# %% Processing functions


def calculate_pushrim_kinetics_biofeedback(
    data: dict[str, ktk.TimeSeries], *, show_plot: bool = False
) -> dict[str, Any]:
    """
    Calculate pushrim kinetics biofeedback.

    Parameters
    ----------
    data
        Dictionary of TimeSeries as returned by NextWheel.fetch().
    show_plot
        Optional. True to show a plot of the processed data.

    Returns
    -------
    dict[str, Any]
        A dictionary with those keys:
        - "Fpeak": the calculated peak force over the last pushes, or 0.0 if
          there were not enough pushes.
        - "Ftot": the Ftot curve, in newton.

    """
    # Default output
    out: dict[str, Any] = {
        "Fpeak": 0.0,
        "FtotCurve": [0 for _ in range(N_POINTS)],
    }

    # Return if no data
    if len(data["Analog"].time) == 0:
        return out

    # Keep the last seconds
    if data["Analog"].time[-1] - TIME_SPAN > data["Analog"].time[0]:
        data["Analog"] = data["Analog"].get_ts_after_time(
            data["Analog"].time[-1] - TIME_SPAN
        )

    # Resample
    data["Analog"].resample(RESAMPLE_FREQUENCY, in_place=True)
    data["Encoder"].resample(
        data["Analog"].time, extrapolate=True, in_place=True
    )
    ts = data["Analog"].merge(data["Encoder"])

    # Process kinetics
    ts = pk.remove_offsets(ts)
    ts.data["Ftot"] = np.sqrt(np.sum(ts.data["Forces"] ** 2, axis=1))
    try:
        ts = ktk.cycles.detect_cycles(
            ts,
            "Ftot",
            event_names=("push", "recovery"),
            thresholds=(10, 5),
            min_durations=(0.1, 0.2),
        )
    except IndexError:  # no cycle detected
        pass

    # Calculate parameters
    Fpeak: list[float] = []
    n_pushes = ts.count_events("push")
    if n_pushes >= N_PUSHES:
        for i_push in range(n_pushes - N_PUSHES, n_pushes):
            subts = ts.get_ts_between_events(
                "push", "recovery", i_push, i_push
            )
            Fpeak.append(np.max(subts.data["Ftot"]))

    if len(Fpeak) == 0:
        out["Fpeak"] = 0.0
    else:
        out["Fpeak"] = np.mean(Fpeak)

    final_frequency = float(N_POINTS) / (ts.time[-1] - ts.time[0])
    ts.resample(final_frequency, in_place=True)

    ts.data["Ftot"][ts.isnan("Ftot")] = 0.0

    out["FtotCurve"] = ts.data["Ftot"].tolist()

    if show_plot:
        ts.plot(["Forces", "Ftot"])

    return out
