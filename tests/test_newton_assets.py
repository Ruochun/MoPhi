from pathlib import Path
import sys
import types

import pytest

from mophi.utils.newton_assets import download_newton_asset


def _install_fake_newton(monkeypatch, download_asset):
    newton_module = types.ModuleType("newton")
    utils_module = types.ModuleType("newton.utils")
    utils_module.download_asset = download_asset
    newton_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "newton", newton_module)
    monkeypatch.setitem(sys.modules, "newton.utils", utils_module)


def test_complete_cache_is_reused(tmp_path, monkeypatch):
    asset_directory = tmp_path / "asset"
    required_file = asset_directory / "model" / "robot.usd"
    required_file.parent.mkdir(parents=True)
    required_file.touch()
    calls = []

    def download_asset(asset_folder, force_refresh=False):
        calls.append((asset_folder, force_refresh))
        return asset_directory

    _install_fake_newton(monkeypatch, download_asset)
    assert download_newton_asset("robot", ["model/robot.usd"]) == asset_directory
    assert calls == [("robot", False)]


def test_incomplete_cache_is_refreshed_once(tmp_path, monkeypatch):
    asset_directory = tmp_path / "asset"
    calls = []

    def download_asset(asset_folder, force_refresh=False):
        calls.append((asset_folder, force_refresh))
        if force_refresh:
            required_file = asset_directory / "config" / "policy.yaml"
            required_file.parent.mkdir(parents=True)
            required_file.touch()
        return asset_directory

    _install_fake_newton(monkeypatch, download_asset)
    assert download_newton_asset("robot", ["config/policy.yaml"]) == asset_directory
    assert calls == [("robot", False), ("robot", True)]


def test_missing_file_after_refresh_is_reported(tmp_path, monkeypatch):
    _install_fake_newton(monkeypatch, lambda asset_folder, force_refresh=False: tmp_path / "asset")
    with pytest.raises(RuntimeError, match="still missing: config/policy.yaml.*last resort"):
        download_newton_asset("robot", ["config/policy.yaml"])


@pytest.mark.parametrize("required_path", ["/absolute/file", "../outside", "model/../../outside"])
def test_required_paths_must_stay_inside_asset_directory(required_path, monkeypatch):
    _install_fake_newton(monkeypatch, lambda asset_folder, force_refresh=False: Path("unused"))
    with pytest.raises(ValueError, match="relative paths"):
        download_newton_asset("robot", [required_path])
