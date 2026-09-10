"""
Main entry point to launch commands from Godot.

This script listens for JSON strings of this form:
    {
        "id": str,
        "command": str,
        "kwargs": dict[str, Any]
    }

where command is a key of COMMAND_MAPPING, kwargs are the arguments sent to
the function in COMMAND_MAPPING, and id is any unique ID, that serves to match
the returned value to its caller.

Command "close" is reserved for closing the python bridge.

The script returns values using JSON strings of this form:
    {
        "id": str,
        "value": Any
    }

where id is the same as it was received, and value is the function's
return value.

"""

print("--------------------")
print(" WheelSims Analysis ")
print("--------------------")
print("Importing Python Modules...")

# ruff: disable[E402]
# because we want the text to appear in the console without delay
import os
import time
import traceback
from collections.abc import Callable
from itertools import cycle

import biofeedback
import biofeedback_pushrim_kinetics as bf_pk
import data_logging
from python_bridge import GODOT_TO_PYTHON_PORT, IP, Receiver, sender

# ruff: enable[E402]

# How many seconds to sleep before polling UDP again once it's empty
SLEEP_TIME_ON_EMPTY_UDP_BUFFER = 1 / 60  # s
SPINNER = cycle("|/-\\")


def _test(arg1: int, arg2: int) -> list:
    """Answer to test command (used in unit tests)."""
    time.sleep(1)
    return [arg1, arg2, 1, 2, 3]


COMMAND_MAPPING: dict[str, Callable] = {
    "test": _test,
    "biofeedback_kinematics": biofeedback.biofeedback_kinematics,
    "biofeedback_stop": biofeedback.biofeedback_stop,
    "biofeedback_pushrim_kinetics_connect": bf_pk.connect,
    "biofeedback_pushrim_kinetics_process": bf_pk.process,
    "start_logging": data_logging.start_log,
    "create_trial": data_logging.create_trial,
    "data_logging": data_logging.save_data,
    "end_trial": data_logging.end_trial,
    "end_logging": data_logging.end_log,
}


if __name__ == "__main__":
    # Create the receiver
    receiver = Receiver(ip=IP, port=GODOT_TO_PYTHON_PORT, timeout=0.0)
    # Send "ready" to Godot
    sender.send({"id": "ready", "value": None})
    print("Ready.")

    stay_in_loop = True
    # Listening Godot requests
    while True:
        # Execute every command in the UDP buffer
        while command_dict := receiver.receive():
            command = command_dict["command"]
            kwargs = command_dict["kwargs"]
            command_id = command_dict["id"]

            if command == "close":
                # Close now
                stay_in_loop = False
                break

            try:
                return_value = COMMAND_MAPPING[command](
                    **command_dict["kwargs"]
                )
            except Exception:
                return_value = None
                print("======================")
                print(f"Exception in command {command} with kwargs {kwargs}.")
                traceback.print_exc()

            # Send back the return value
            sender.send({"id": command_id, "value": return_value})
            print(next(SPINNER), end="\r", flush=True)

        # Do not execute repeating commands after shutdown request
        if not stay_in_loop:
            break

        # Wait some time before checking if a new request was received
        time.sleep(SLEEP_TIME_ON_EMPTY_UDP_BUFFER)

    # Quit
    os._exit(0)
