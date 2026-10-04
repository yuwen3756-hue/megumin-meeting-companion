"""Telemetry privacy contracts; no GPU, native inference, or network is used.

Tests that exercise the installed package run in fresh Python processes because
the application deliberately patches process-global methods and the upstream
Telemetry class is a singleton. Nothing leaks into the pytest process.
"""

from __future__ import annotations

import builtins
import importlib.util
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from meeting.telemetry import disable_openvino_telemetry


ROOT = Path(__file__).resolve().parents[1]
HAS_TELEMETRY = importlib.util.find_spec("openvino_telemetry") is not None


def run_isolated(source: str) -> None:
    result = subprocess.run(
        [sys.executable, "-B", "-c", textwrap.dedent(source)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not HAS_TELEMETRY, reason="Optional OpenVINO telemetry package is absent")
def test_installed_package_declines_without_profile_access_and_discards_forced_events():
    run_isolated(
        """
        import os
        import sys
        from contextlib import ExitStack
        from unittest.mock import patch
        from meeting.telemetry import disable_openvino_telemetry
        from openvino_telemetry import Telemetry
        from openvino_telemetry.backend import backend_ga4
        from openvino_telemetry.utils.opt_in_checker import ConsentCheckResult, OptInChecker

        assert Telemetry.get_version() == '2025.2.0', 'Revalidate the guard against the changed package'
        environment_before = dict(os.environ)
        forbidden = AssertionError('Unexpected profile, filesystem, or network access')
        with ExitStack() as stack:
            # OS labeling is unrelated to consent/sending; on Windows its
            # discovery can spawn `ver` and open a pipe via io.open.
            stack.enter_context(patch.object(backend_ga4, 'system', return_value='Windows'))
            # Forced-event formatting probes Linux /proc even on Windows.
            # Isolate that metadata probe while exercising the actual sender.
            stack.enter_context(patch.object(backend_ga4, 'is_docker', return_value=False))
            blocked = [
                stack.enter_context(patch(target, side_effect=forbidden))
                for target in (
                    'builtins.open', 'io.open', 'os.mkdir', 'os.makedirs',
                    'os.remove', 'os.unlink', 'socket.socket.connect',
                    'socket.socket.sendto', 'urllib.request.urlopen',
                )
            ]
            blocked += [
                stack.enter_context(patch.object(owner, name, side_effect=forbidden))
                for owner, name in (
                    (OptInChecker, 'consent_file_base_dir'),
                    (OptInChecker, 'update_result'),
                    (backend_ga4.GA4Backend, 'generate_new_cid_file'),
                    (Telemetry, 'get_stats'),
                )
            ]
            assert disable_openvino_telemetry() is True
            assert disable_openvino_telemetry() is True
            assert OptInChecker().check(False, False) == ConsentCheckResult.DECLINED
            telemetry = Telemetry(
                app_name='MeguminMeet-test', app_version='test', tid='test',
                backend='ga4', enable_opt_in_dialog=False, disable_in_ci=False,
            )
            assert telemetry.consent is False
            try:
                with patch.object(telemetry.sender.executor, 'submit', side_effect=forbidden) as submit:
                    telemetry.send_event('test', 'event', 'ordinary')
                    telemetry.send_event('test', 'event', 'forced', force_send=True)
                    # This verifies the sender independently of consent gating.
                    telemetry.sender.send(telemetry.backend, {'event': 'direct'})
                    submit.assert_not_called()
                assert telemetry.sender.queue_size == 0
            finally:
                telemetry.sender.executor.shutdown(wait=False)
            for operation in blocked:
                operation.assert_not_called()
        assert dict(os.environ) == environment_before
        assert 'openvino' not in sys.modules
        assert 'openvino_genai' not in sys.modules
        """
    )


@pytest.mark.parametrize("missing_name", ["openvino_telemetry", "unexpected_dependency"])
def test_optional_absence_is_allowed_but_broken_dependency_fails_closed(monkeypatch, missing_name):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name.startswith("openvino_telemetry"):
            raise ModuleNotFoundError("Synthetic missing module", name=missing_name)
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", missing)
        if missing_name == "openvino_telemetry":
            assert disable_openvino_telemetry() is False
        else:
            with pytest.raises(RuntimeError, match="Cannot establish"):
                disable_openvino_telemetry()
    assert builtins.__import__ is original_import


@pytest.mark.skipif(not HAS_TELEMETRY, reason="Optional OpenVINO telemetry package is absent")
def test_changed_installed_api_fails_closed():
    run_isolated(
        """
        from unittest.mock import patch
        from meeting.telemetry import disable_openvino_telemetry
        from openvino_telemetry.utils.opt_in_checker import OptInChecker

        with patch.object(OptInChecker, 'check', None):
            try:
                disable_openvino_telemetry()
            except RuntimeError as exc:
                assert 'incompatible' in str(exc)
            else:
                raise AssertionError('An incompatible telemetry API was silently accepted')
        """
    )


def test_offline_entry_point_installs_guard_without_affecting_pytest_process():
    run_isolated(
        """
        import os
        import sys
        from unittest.mock import patch
        from meeting import offline

        with patch.object(offline, 'disable_openvino_telemetry', return_value=True) as guard:
            offline.enable()
            guard.assert_called_once_with()
        assert os.environ['HF_HUB_OFFLINE'] == '1'
        assert 'openvino' not in sys.modules
        assert 'openvino_genai' not in sys.modules
        """
    )
