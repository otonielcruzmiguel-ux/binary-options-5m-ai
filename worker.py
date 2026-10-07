import asyncio, json, sqlite3, time
from pathlib import Path
import joblib, pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from src.deriv_market import candles
from src.history_store import upsert, load, count
from src.binary5m import FEATURES, make_features
from src.investing_calendar import refresh as refresh_news, block as news_block

MARKETS={"EUR/USD":"frxEURUSD","GBP/USD":"frxGBPUSD","USD/JPY":"frxUSDJPY","AUD/USD":"frxAUDUSD","USD/CAD":"frxUSDCAD","USD/CHF":"frxUSDCHF","EUR/JPY":"frxEURJPY","GBP/JPY":"frxGBPJPY","EUR/GBP":"frxEURGBP","AUD/JPY":"frxAUDJPY"}
DATA=Path("/data"); DATA.mkdir(parents=True,exist_ok=True)
DB=DATA/"learning.db"; THRESHOLD=max(1/(1+.80)+.03,.58)
NEWS_REFRESH_SECONDS=900
_last_news_refresh=0
STRATEGY_FIELDS=["breakout_up","breakout_down","retest_up","retest_down","support_bounce","resistance_bounce","ema_trend","ema_cross","rsi_support","rsi_resistance","rsi_bull_div","rsi_bear_div","bb_rsi_buy","bb_rsi_sell","trend_pullback","confluence"]

def connect():
    c=sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS signals(market TEXT,bucket TEXT,expires TEXT,decision TEXT,entry REAL,exit REAL,confidence REAL,result TEXT,PRIMARY KEY(market,bucket))")
    cols={r[1] for r in c.execute("PRAGMA table_info(signals)")}
    if "context" not in cols: c.execute("ALTER TABLE signals ADD COLUMN context TEXT")
    c.execute("CREATE TABLE IF NOT EXISTS summaries(period TEXT PRIMARY KEY,created TEXT,evaluated INTEGER,wins INTEGER,losses INTEGER,ties INTEGER,accuracy REAL)")
    c.execute("""CREATE TABLE IF NOT EXISTS evolution(
      period TEXT PRIMARY KEY, created TEXT, evaluated INTEGER, wins INTEGER, losses INTEGER,
      accuracy REAL, change_type TEXT, detail TEXT)""")
    c.commit(); return c

def train(raw):
    d=make_features(raw,5,True).dropna(subset=FEATURES+["target"])
    if len(d)<500: return None
    cut=int(len(d)*.70); tr=d.iloc[:cut]; te=d.iloc[cut:]
    candidate=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    candidate.fit(tr[FEATURES],tr["target"].astype(int))
    p=candidate.predict_proba(te[FEATURES])[:,1]; y=te["target"].astype(int).to_numpy()
    take=(p>=THRESHOLD)|(p<=1-THRESHOLD); pred=(p>=.5).astype(int)
    signal_acc=float((pred[take]==y[take]).mean()) if take.any() else None
    signals=int(take.sum())
    # Producción se ajusta con todo el histórico solo después de medir el tramo futuro.
    m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    m.fit(d[FEATURES],d["target"].astype(int))
    return {"model":m,"features":FEATURES,"threshold":THRESHOLD,"payout":.80,"trained_at":pd.Timestamp.now(tz="UTC").isoformat(),"validation_accuracy":signal_acc,"validation_signals":signals}

def adaptive_threshold(con,market,now):
    # Solo adapta con muestra forward suficiente; nunca por una pérdida aislada.
    since=(now-pd.Timedelta(days=7)).isoformat()
    rows=con.execute("SELECT result FROM signals WHERE market=? AND bucket>=? AND result IN ('GANADA','PERDIDA')",(market,since)).fetchall()
    n=len(rows)
    if n<30: return THRESHOLD,n,None
    wins=sum(1 for r in rows if r[0]=="GANADA"); acc=wins/n
    # Si el rendimiento reciente cae, exige más confianza; si es sólido, relaja muy poco.
    adj=.03 if acc<.56 else (.015 if acc<.60 else (-.005 if acc>=.66 and n>=60 else 0))
    return min(.68,max(THRESHOLD,THRESHOLD+adj)),n,acc

def hourly_evolution(con,now):
    period=now.floor("1h").isoformat()
    if con.execute("SELECT 1 FROM evolution WHERE period=?",(period,)).fetchone(): return
    start=now.floor("1h")-pd.Timedelta(hours=1)
    rows=con.execute("SELECT result FROM signals WHERE bucket>=? AND bucket<? AND result IN ('GANADA','PERDIDA')",(start.isoformat(),now.floor("1h").isoformat())).fetchall()
    n=len(rows); w=sum(r[0]=="GANADA" for r in rows); l=sum(r[0]=="PERDIDA" for r in rows)
    acc=(w/n) if n else None
    # Esta bitácora documenta la evaluación. No afirma cambios de algoritmo que no hayan ocurrido.
    detail=("Sin muestra suficiente; se mantiene la estrategia actual." if n<10 else
            ("Rendimiento horario bajo; los filtros adaptativos pueden exigir mayor confianza." if acc is not None and acc<.56 else
             "Rendimiento horario estable; sin cambio automático de estrategia."))
    change_type="EVALUACIÓN HORARIA"
    con.execute("INSERT OR IGNORE INTO evolution VALUES(?,?,?,?,?,?,?,?)",(period,now.isoformat(),n,w,l,acc,change_type,detail))
    con.commit()
    print(f"EVOLUCION_1H evaluadas={n} ganadas={w} perdidas={l} accuracy={'NA' if acc is None else f'{acc:.3f}'} detalle={detail}",flush=True)

def context_of(row):
    out={}
    for k in STRATEGY_FIELDS:
        if k in row.index and pd.notna(row[k]): out[k]=round(float(row[k]),5)
    return json.dumps(out,separators=(",",":"))

def summarize(con,now):
    period=now.floor("8h").isoformat()
    if con.execute("SELECT 1 FROM summaries WHERE period=?",(period,)).fetchone(): return
    start=now.floor("8h")-pd.Timedelta(hours=8)
    rows=con.execute("SELECT result FROM signals WHERE bucket>=? AND bucket<?",(start.isoformat(),now.floor("8h").isoformat())).fetchall()
    vals=[r[0] for r in rows if r[0] in ("GANADA","PERDIDA","EMPATE")]
    w=vals.count("GANADA"); l=vals.count("PERDIDA"); t=vals.count("EMPATE"); n=w+l
    acc=w/n if n else None
    con.execute("INSERT OR IGNORE INTO summaries VALUES(?,?,?,?,?,?,?)",(period,now.isoformat(),len(vals),w,l,t,acc))
    con.commit()
    print(f"RESUMEN_8H evaluadas={len(vals)} ganadas={w} perdidas={l} empates={t} accuracy={'NA' if acc is None else f'{acc:.3f}'}",flush=True)

async def cycle():
    global _last_news_refresh
    con=connect(); now=pd.Timestamp.now(tz="UTC")
    if time.time()-_last_news_refresh>=NEWS_REFRESH_SECONDS:
        news=refresh_news(); _last_news_refresh=time.time()
        print(f"NOTICIAS Investing.com: {'actualizadas '+str(len(news)) if news is not None else 'no disponible; se conserva el último estado local'}",flush=True)
    for market,symbol in MARKETS.items():
        try:
            # Deriv solo alimenta el almacén; entrenamiento y evaluación leen de /data.
            have=count(symbol)
            fetch_n=10000 if have<10000 else 120
            fresh=await candles(symbol,count=fetch_n,granularity=60)
            upsert(symbol,fresh)
            raw=load(symbol,100000)
            if raw.empty: continue
            path=DATA/f"model_{symbol}.joblib"; meta=DATA/f"model_{symbol}.json"
            retrain=not path.exists()
            if meta.exists():
                try: retrain=(now-pd.Timestamp(json.loads(meta.read_text())["trained_at"])).total_seconds()>=1800
                except Exception: retrain=True
            if retrain:
                payload=train(raw)
                if payload:
                    tmp=DATA/f"model_{symbol}.tmp"; joblib.dump(payload,tmp); tmp.replace(path)
                    meta.write_text(json.dumps({"trained_at":payload["trained_at"],"validation_accuracy":payload.get("validation_accuracy"),"validation_signals":payload.get("validation_signals")}))
                    print(f"{market}: modelo reentrenado",flush=True)
            if not path.exists(): continue
            payload=joblib.load(path)
            feat=make_features(raw,5,False).dropna(subset=FEATURES)
            if feat.empty: continue
            latest=feat.iloc[-1]; bucket=now.floor("5min")
            p=float(payload["model"].predict_proba(feat[payload["features"]].iloc[[-1]])[0,1])
            live_threshold,live_n,live_acc=adaptive_threshold(con,market,now)
            decision="SUBE" if p>=live_threshold else ("BAJA" if p<=1-live_threshold else "NO OPERAR")
            base,quote=market.split("/")
            blocked,news_reason,news_event=news_block(now,(base,quote),before=15,after=30)
            if blocked:
                decision="NO OPERAR"
                print(f"NO_OPERAR_NOTICIA market={market} motivo={news_reason}",flush=True)
            if decision!="NO OPERAR":
                ctx=json.loads(context_of(latest)); ctx["news_status"]="CLEAR"; ctx["news_source"]="Investing.com"; ctx=json.dumps(ctx,separators=(",",":"))
                con.execute("INSERT OR IGNORE INTO signals(market,bucket,expires,decision,entry,exit,confidence,result,context) VALUES(?,?,?,?,?,?,?,?,?)",(market,bucket.isoformat(),(bucket+pd.Timedelta(minutes=5)).isoformat(),decision,float(raw.iloc[-1].close),None,max(p,1-p),"PENDIENTE",ctx))
            for m,b,e,d,entry,ctx in con.execute("SELECT market,bucket,expires,decision,entry,context FROM signals WHERE market=? AND result='PENDIENTE'",(market,)).fetchall():
                exp=pd.Timestamp(e)
                if now>=exp:
                    after=raw[raw["timestamp"]>=exp]
                    if not after.empty:
                        exitp=float(after.iloc[0].close); won=(d=="SUBE" and exitp>entry) or (d=="BAJA" and exitp<entry)
                        result="EMPATE" if exitp==entry else ("GANADA" if won else "PERDIDA")
                        con.execute("UPDATE signals SET exit=?,result=? WHERE market=? AND bucket=?",(exitp,result,m,b))
                        if result=="PERDIDA": print(f"FALLO market={m} bucket={b} decision={d} contexto={ctx}",flush=True)
            con.commit()
        except Exception as e: print(f"{market}: {e}",flush=True)
    hourly_evolution(con,now); summarize(con,now); con.close()

if __name__=="__main__":
    print("Worker de aprendizaje 24/7 iniciado",flush=True)
    while True:
        try: asyncio.run(cycle())
        except Exception as e: print(f"worker: {e}",flush=True)
        time.sleep(30)
