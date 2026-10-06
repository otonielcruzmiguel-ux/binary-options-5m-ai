import asyncio, json, sqlite3, time
from pathlib import Path
import joblib, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from src.deriv_market import candles
from src.binary5m import FEATURES, make_features

MARKETS={"EUR/USD":"frxEURUSD","GBP/USD":"frxGBPUSD","USD/JPY":"frxUSDJPY","AUD/USD":"frxAUDUSD"}
DATA=Path("/data"); DATA.mkdir(parents=True,exist_ok=True)
DB=DATA/"learning.db"; THRESHOLD=max(1/(1+.80)+.03,.58)

def connect():
    c=sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS signals(market TEXT,bucket TEXT,expires TEXT,decision TEXT,entry REAL,exit REAL,confidence REAL,result TEXT,PRIMARY KEY(market,bucket))")
    c.commit(); return c

def train(raw):
    d=make_features(raw,5,True).dropna(subset=FEATURES+["target"])
    if len(d)<500: return None
    m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    m.fit(d[FEATURES],d["target"].astype(int))
    return {"model":m,"features":FEATURES,"threshold":THRESHOLD,"payout":.80,"trained_at":pd.Timestamp.now(tz="UTC").isoformat()}

async def cycle():
    con=connect(); now=pd.Timestamp.now(tz="UTC")
    for market,symbol in MARKETS.items():
        try:
            raw=await candles(symbol,count=5000,granularity=60)
            path=DATA/f"model_{symbol}.joblib"; meta=DATA/f"model_{symbol}.json"
            retrain=not path.exists()
            if meta.exists():
                try: retrain=(now-pd.Timestamp(json.loads(meta.read_text())["trained_at"])).total_seconds()>=1800
                except Exception: retrain=True
            if retrain:
                payload=train(raw)
                if payload:
                    tmp=DATA/f"model_{symbol}.tmp"
                    joblib.dump(payload,tmp); tmp.replace(path)
                    meta.write_text(json.dumps({"trained_at":payload["trained_at"]}))
                    print(f"{market}: modelo reentrenado",flush=True)
            if not path.exists(): continue
            payload=joblib.load(path)
            feat=make_features(raw,5,False).dropna(subset=FEATURES)
            if feat.empty: continue
            bucket=now.floor("5min")
            p=float(payload["model"].predict_proba(feat[payload["features"]].iloc[[-1]])[0,1])
            decision="SUBE" if p>=THRESHOLD else ("BAJA" if p<=1-THRESHOLD else "NO OPERAR")
            if decision!="NO OPERAR":
                con.execute("INSERT OR IGNORE INTO signals VALUES(?,?,?,?,?,?,?,?)",(market,bucket.isoformat(),(bucket+pd.Timedelta(minutes=5)).isoformat(),decision,float(raw.iloc[-1].close),None,max(p,1-p),"PENDIENTE"))
            for m,b,e,d,entry in con.execute("SELECT market,bucket,expires,decision,entry FROM signals WHERE market=? AND result='PENDIENTE'",(market,)).fetchall():
                exp=pd.Timestamp(e)
                if now>=exp:
                    after=raw[raw["timestamp"]>=exp]
                    if not after.empty:
                        exitp=float(after.iloc[0].close)
                        won=(d=="SUBE" and exitp>entry) or (d=="BAJA" and exitp<entry)
                        result="EMPATE" if exitp==entry else ("GANADA" if won else "PERDIDA")
                        con.execute("UPDATE signals SET exit=?,result=? WHERE market=? AND bucket=?",(exitp,result,m,b))
            con.commit()
        except Exception as e: print(f"{market}: {e}",flush=True)
    con.close()

if __name__=="__main__":
    print("Worker de aprendizaje 24/7 iniciado",flush=True)
    while True:
        try: asyncio.run(cycle())
        except Exception as e: print(f"worker: {e}",flush=True)
        time.sleep(30)
