"""Disable OpenVINO's Python telemetry only in the current process."""


def disable_openvino_telemetry() -> bool:
    """Call before importing OpenVINO, including hardware discovery imports.

    Verified against openvino-telemetry 2025.2.0: OpenVINO's conversion module
    is imported by its top-level package and initializes telemetry without an
    opt-in dialog. A missing consent file then counts as consent. DO_NOT_TRACK
    is not consulted. Declining the in-memory check avoids consent/client-ID/
    usage-file access; disabling the sender also drops forced telemetry events.

    No opt-out command, user-file edits, environment changes, or OpenVINO
    runtime imports occur here. Missing telemetry is valid for NVIDIA-only
    environments. An incompatible installed telemetry API is an error rather
    than a silent loss of the guard. Call again safely for each entry point.
    """
    try:
        from openvino_telemetry.utils.opt_in_checker import ConsentCheckResult, OptInChecker
        from openvino_telemetry.utils.sender import TelemetrySender
    except ModuleNotFoundError as exc:
        if exc.name == "openvino_telemetry":
            return False
        raise RuntimeError("Cannot establish the process-local OpenVINO telemetry guard.") from exc
    except ImportError as exc:
        raise RuntimeError("The installed OpenVINO telemetry API is incompatible with its offline guard.") from exc

    if not callable(getattr(OptInChecker, "check", None)) or not callable(getattr(TelemetrySender, "send", None)):
        raise RuntimeError("The installed OpenVINO telemetry API is incompatible with its offline guard.")
    declined = ConsentCheckResult.DECLINED

    def discard_event(*args, **kwargs):
        return None

    def decline_consent(*args, **kwargs):
        return declined

    # The sender override covers force_send=True as well as ordinary events.
    # Apply it first, before replacing the consent check. Neither import above
    # constructs the telemetry singleton or starts an OpenVINO runtime.
    TelemetrySender.send = discard_event
    OptInChecker.check = decline_consent
    return True
