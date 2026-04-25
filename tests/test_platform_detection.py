"""Smoke tests for benchmark.metrics.collectors.detect_platform.

Platform detection touches /etc/nv_tegra_release, /proc/device-tree/model,
/dev/hailo*, hailortcli, and lspci. We mock the relevant probes rather than
relying on the host hardware so the tests work on dev machines too.
"""
from unittest.mock import MagicMock, patch

from benchmark.metrics.collectors import detect_platform
from benchmark.schemas import Platform


def test_returns_platform_enum_on_real_host():
    # On any host, detect_platform() must return a Platform enum value.
    result = detect_platform()
    assert isinstance(result, Platform)


def test_jetson_detected_when_nv_tegra_release_present():
    with patch("benchmark.metrics.collectors.Path") as mock_path:
        mock_path.return_value.exists.return_value = True
        assert detect_platform() == Platform.JETSON_ORIN_NANO


def test_falls_back_to_jetson_when_no_signals():
    # No /etc/nv_tegra_release, no /proc/device-tree/model.
    with patch("benchmark.metrics.collectors.Path") as mock_path:
        mock_path.return_value.exists.return_value = False
        with patch("builtins.open", side_effect=FileNotFoundError):
            assert detect_platform() == Platform.JETSON_ORIN_NANO


def test_rpi_ai_hat_plus_2_detected_via_hailortcli():
    fake_dev_path = MagicMock()
    fake_dev_path.exists.return_value = False
    fake_dev_path.glob.return_value = [MagicMock()]  # at least one /dev/hailo*

    with patch("benchmark.metrics.collectors.Path") as mock_path:
        mock_path.return_value = fake_dev_path
        with patch(
            "builtins.open",
            new=MagicMock(
                return_value=MagicMock(
                    __enter__=lambda self: self,
                    __exit__=lambda *_: None,
                    read=lambda: "Raspberry Pi 5 Model B",
                )
            ),
        ):
            with patch("benchmark.metrics.collectors.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    stdout="Identifying board\nDevice: Hailo-10H\n",
                    returncode=0,
                )
                assert detect_platform() == Platform.RPI_AI_HAT_PLUS_2
