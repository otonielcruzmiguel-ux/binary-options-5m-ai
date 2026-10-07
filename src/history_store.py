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
