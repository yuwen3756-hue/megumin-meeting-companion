"""Process-local network guard; does not change Windows security settings."""
import os
import socket

from .telemetry import disable_openvino_telemetry


def enable():
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
        os.environ[key] = "1"
    def denied(*args, **kwargs):
        raise OSError("MeguminMeet meeting mode is offline. Use Setup.cmd for explicit model downloads.")
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.socket.sendto = denied
    socket.create_connection = denied
    socket.getaddrinfo = denied
    disable_openvino_telemetry()
