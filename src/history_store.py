import sqlite3
from pathlib import Path
import pandas as pd

DATA=Path("/data"); DATA.mkdir(parents=True,exist_ok=True)
DB=DATA/"market_history.db"

def connect():
    c=sqlite3.connect(DB,timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS candles(
      symbol TEXT NOT NULL, timestamp TEXT NOT NULL, open REAL, high REAL, low REAL, close REAL, volume REAL DEFAULT 0,
      PRIMARY KEY(symbol,timestamp))""")
    c.commit(); return c

def upsert(symbol,df):
    if df is None or df.empty:return 0
    rows=[(symbol,pd.Timestamp(r.timestamp).isoformat(),float(r.open),float(r.high),float(r.low),float(r.close),float(getattr(r,"volume",0))) for r in df.itertuples()]
    with connect() as c:
        c.executemany("INSERT OR REPLACE INTO candles VALUES(?,?,?,?,?,?,?)",rows)
    return len(rows)

def load(symbol,limit=10000):
    with connect() as c:
        d=pd.read_sql_query("SELECT timestamp,open,high,low,close,volume FROM candles WHERE symbol=? ORDER BY timestamp DESC LIMIT ?",c,params=(symbol,int(limit)))
    if d.empty:return d
    d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
    return d.sort_values("timestamp").reset_index(drop=True)

def count(symbol):
    with connect() as c:return c.execute("SELECT COUNT(*) FROM candles WHERE symbol=?",(symbol,)).fetchone()[0]

# Provider-aware storage. Legacy Deriv candles remain in the original table.
def _connect_scoped():
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS candles_scoped(
        provider TEXT NOT NULL, market_type TEXT NOT NULL,
        symbol TEXT NOT NULL, timestamp TEXT NOT NULL,
        open REAL, high REAL, low REAL, close REAL, volume REAL DEFAULT 0,
        PRIMARY KEY(provider,market_type,symbol,timestamp))""")
    c.commit()
    return c

def _scope(provider, market_type):
    provider = str(provider).strip().lower()
    market_type = str(market_type).strip().lower()
    if provider not in ("deriv", "iqoption"):
        raise ValueError("Unsupported market data provider")
    if market_type not in ("regular", "otc"):
        raise ValueError("Unsupported market type")
    return provider, market_type

def upsert_scoped(provider, market_type, symbol, df):
    provider, market_type = _scope(provider, market_type)
    if df is None or df.empty:
        return 0
    rows = [(provider, market_type, symbol, pd.Timestamp(r.timestamp).isoformat(),
             float(r.open), float(r.high), float(r.low), float(r.close),
             float(getattr(r, "volume", 0))) for r in df.itertuples()]
    c = _connect_scoped()
    try:
        with c:
            c.executemany("""INSERT OR REPLACE INTO candles_scoped
                (provider,market_type,symbol,timestamp,open,high,low,close,volume)
                VALUES (?,?,?,?,?,?,?,?,?)""", rows)
    finally:
        c.close()
    return len(rows)

def load_scoped(provider, market_type, symbol, limit=10000):
    provider, market_type = _scope(provider, market_type)
    c = _connect_scoped()
    try:
        d = pd.read_sql_query("""SELECT timestamp,open,high,low,close,volume
            FROM candles_scoped WHERE provider=? AND market_type=? AND symbol=?
            ORDER BY timestamp DESC LIMIT ?""", c,
            params=(provider, market_type, symbol, int(limit)))
    finally:
        c.close()
    if not d.empty:
        d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
        d = d.sort_values("timestamp").reset_index(drop=True)
    return d
