"""Offline regression tests for quarantine recovery. No trades or external data."""
import sqlite3
import pandas as pd
from worker import pair_quarantine

def test_recovered_pair_requires_fresh_operational_sample():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE signals(market TEXT,bucket TEXT,result TEXT)")
    c.execute("CREATE TABLE rejected_signals(horizon INTEGER,market TEXT,bucket TEXT,result TEXT)")
    now = pd.Timestamp("2026-10-12T12:00:00Z")
    old = (now-pd.Timedelta(days=1)).isoformat()
    c.executemany("INSERT INTO signals VALUES(?,?,?)", [("EUR/USD",old,"PERDIDA")]*80)
    c.execute("CREATE TABLE quarantine_state(horizon INTEGER,market TEXT,blocked INTEGER,changed TEXT,reason TEXT,PRIMARY KEY(horizon,market))")
    recovered=(now-pd.Timedelta(hours=1)).isoformat()
    c.execute("INSERT INTO quarantine_state VALUES(5,'EUR/USD',0,?,'recuperacion sombra n=80 accuracy=0.650')",(recovered,))
    assert pair_quarantine(c,"EUR/USD",now,"signals",80)[0] is True
    c.executemany("INSERT INTO signals VALUES(?,?,?)", [("EUR/USD",(now-pd.Timedelta(minutes=1)).isoformat(),"PERDIDA")]*80)
    assert pair_quarantine(c,"EUR/USD",now,"signals",80)[0] is False
    c.close()
