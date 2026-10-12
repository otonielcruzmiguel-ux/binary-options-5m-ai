"""Offline tests: no IQ Option connection or trading."""
import pandas as pd
import pytest
from src import history_store as store

def test_scoped_history_isolation(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DB", tmp_path / "history.db")
    df = pd.DataFrame([{"timestamp":"2026-10-12T00:00:00Z","open":1.0,"high":1.1,"low":0.9,"close":1.05,"volume":1}])
    assert store.upsert_scoped("iqoption", "regular", "EURUSD", df) == 1
    assert store.load_scoped("iqoption", "otc", "EURUSD").empty
    assert store.load_scoped("deriv", "regular", "EURUSD").empty
    assert len(store.load_scoped("iqoption", "regular", "EURUSD")) == 1
    assert store.upsert_scoped("iqoption", "regular", "EURUSD", df) == 1
    assert len(store.load_scoped("iqoption", "regular", "EURUSD")) == 1

def test_invalid_scope_rejected():
    with pytest.raises(ValueError):
        store._scope("unknown", "regular")
    with pytest.raises(ValueError):
        store._scope("iqoption", "unknown")
