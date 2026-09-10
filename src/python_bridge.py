"""
Python bridge module.

Enables asynchronous, bidirectional UDP communication with Godot to:
- Receive and process incoming JSON commands
- Allow independent modules to send data back to Godot

"""

__author__ = "Mobility and Adaptive System Research Lab"
__copyright__ = """
    Copyright (C) 2026 Mobility and Adaptive System Research Lab
    """
__email__ = "chenier.felix@uqam.ca"
__license__ = "Apache 2.0"


import json
import socket
from typing import Any

IP = "127.0.0.1"
GODOT_TO_PYTHON_PORT = 4243
PYTHON_TO_GODOT_PORT = 4242
DATA_SIZE = 1024  # bytes


class Receiver:
    """
    UDP Receiver.

    Attributes
    ----------
    ip
        IP address to listen to.
    port
        Port to listen to.
    timeout
        Number of seconds before timing out. Use 0.0 for continuous polling.

    """

    def __init__(self, ip: str, port: int, timeout: float):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.settimeout(timeout)
        self.socket.bind((ip, port))

    def receive(self) -> dict[str, Any] | None:
        """
        Poll the receiver for new data.

        Returns
        -------
        dict
            A dictionary with this form:
            command: str          # Function to call
            args: dict[str, Any]  # Keyword args to be sent to the function
            run_mode: str         # Either "once", "start" or "stop"

        """
        try:
            message, address = self.socket.recvfrom(DATA_SIZE)
            return json.loads(message.decode("utf-8"))
        except BlockingIOError:
            return None
        except ConnectionResetError:
            return None
        except TimeoutError:
            return None


class Sender:
    """
    UDP Sender.

    Attributes
    ----------
    ip
        IP address to send to.
    port
        Port to send to.

    """

    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(
        self,
        data: dict[str, Any],
    ) -> None:
        """
        Send data.

        Parameters
        ----------
        command
            Name of the function that sends the data
        data
            Return value of the function

        """
        json_message = json.dumps(data).encode("utf-8")
        self.socket.sendto(json_message, (self.ip, self.port))


# Create a sender instance to be imported by the different modules
sender = Sender(ip=IP, port=PYTHON_TO_GODOT_PORT)
