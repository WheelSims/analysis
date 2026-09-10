"""
Tests for python_bridge module.

"""

import os
import sys
import pytest
import subprocess


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "src"))
sys.path.append(root_dir)


import python_bridge


def test_python_bridge():
    """
    Test python bridge.

    Launches an instance of main.py, sends a test command using UDP, waits
    for a response using UDP, and checks it the response is as expected.

    """
    # Create test receiver and sender
    receiver = python_bridge.Receiver(
        ip=python_bridge.IP,
        port=python_bridge.PYTHON_TO_GODOT_PORT,
        timeout=5.0,
    )
    sender = python_bridge.Sender(
        ip=python_bridge.IP, port=python_bridge.GODOT_TO_PYTHON_PORT
    )

    # Launch Python Bridge
    process = subprocess.Popen(
        [sys.executable, root_dir + "/src/main.py"],
        start_new_session=True,
    )

    try:
        received_data = receiver.receive()
        assert received_data["id"] == "ready"
        assert received_data["value"] == None

        # Check that we can send a test command
        sender.send(
            {
                "command": "test",
                "kwargs": {"arg1": "test", "arg2": 123.45},
                "id": "test_123_456",
            }
        )

        # Check that we receive something
        received_data = receiver.receive()
        assert received_data["id"] == "test_123_456"
        assert received_data["value"] == ["test", 123.45, 1, 2, 3]

        # Close the bridge
        sender.send({"command": "close", "kwargs": {}, "id": "close"})

    except Exception as e:
        raise e
    finally:
        # Close Python Bridge
        process.kill()
        pass

    return
    # Make the function crash in case of error, for example by using assert x == y


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
