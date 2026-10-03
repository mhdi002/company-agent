from core.config import deep_merge, load_config
from model.config import preset


def test_load_config_defaults():
    cfg = load_config()
    assert cfg["srlm"]["K"] == 8
    assert cfg["srlm"]["use_subcalls"] is False
    assert cfg["srlm"]["step_timeout_s"] == 60


def test_env_override(monkeypatch):
    monkeypatch.setenv("PA__SRLM__K", "4")
    monkeypatch.setenv("PA__AGENT__POLICY", "template")
    cfg = load_config()
    assert cfg["srlm"]["K"] == 4 and cfg["agent"]["policy"] == "template"


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_1b_budget():
    n = preset("1b", 48000).param_count()
    assert 0.95e9 < n < 1.1e9
