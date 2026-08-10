import pytest

from project_config import env_bool, env_int, env_path


def test_environment_helpers(monkeypatch, tmp_path):
    monkeypatch.setenv("FLAG", "yes")
    monkeypatch.setenv("COUNT", "7")
    monkeypatch.setenv("TARGET", str(tmp_path))

    assert env_bool("FLAG", False) is True
    assert env_int("COUNT", 1, minimum=1) == 7
    assert env_path("TARGET", "fallback") == tmp_path


def test_environment_helpers_reject_invalid_values(monkeypatch):
    monkeypatch.setenv("FLAG", "sometimes")
    monkeypatch.setenv("COUNT", "0")

    with pytest.raises(ValueError, match="FLAG"):
        env_bool("FLAG", False)
    with pytest.raises(ValueError, match="at least 1"):
        env_int("COUNT", 1, minimum=1)
