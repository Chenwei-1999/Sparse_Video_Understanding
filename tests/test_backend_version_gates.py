import pytest

from verl.workers.rollout.replica import _require_pkg_version, RolloutReplicaRegistry


def test_require_pkg_version_error_message(monkeypatch):
    def fake_version(pkg):
        if pkg == "vllm":
            return "0.8.0"
        return "0.0.0"

    monkeypatch.setattr("verl.workers.rollout.replica.version", fake_version)

    with pytest.raises(ImportError) as exc:
        _require_pkg_version("vllm", "0.11.0", "Install a compatible version")

    msg = str(exc.value)
    assert "vllm 0.8.0 is too old" in msg
    assert "Install a compatible version" in msg


def test_registry_unknown_backend():
    with pytest.raises(ValueError):
        RolloutReplicaRegistry.get("nonexistent")

