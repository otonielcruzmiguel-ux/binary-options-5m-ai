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
DB=DATA/"learning.db"; PAYOUT=.80; BREAK_EVEN=1/(1+PAYOUT); THRESHOLD=max(BREAK_EVEN+.03,.58)
NEWS_REFRESH_SECONDS=900
_last_news_refresh=0
STRATEGY_FIELDS=["breakout_up","breakout_down","retest_up","retest_down","support_bounce","resistance_bounce","ema_trend","ema_cross","rsi_support","rsi_resistance","rsi_bull_div","rsi_bear_div","bb_rsi_buy","bb_rsi_sell","trend_pullback","confluence"]

def connect():
    c=sqlite3.connect(DB)
    c.execute("CREATE TABLE IF NOT EXISTS signals(market TEXT,bucket TEXT,expires TEXT,decision TEXT,entry REAL,exit REAL,confidence REAL,result TEXT,PRIMARY KEY(market,bucket))")
    c.execute("CREATE TABLE IF NOT EXISTS signals_1m(market TEXT,bucket TEXT,expires TEXT,decision TEXT,entry REAL,exit REAL,confidence REAL,result TEXT,context TEXT,PRIMARY KEY(market,bucket))")
    cols={r[1] for r in c.execute("PRAGMA table_info(signals)")}
    if "context" not in cols: c.execute("ALTER TABLE signals ADD COLUMN context TEXT")
    c.execute("CREATE TABLE IF NOT EXISTS summaries(period TEXT PRIMARY KEY,created TEXT,evaluated INTEGER,wins INTEGER,losses INTEGER,ties INTEGER,accuracy REAL)")
    c.execute("""CREATE TABLE IF NOT EXISTS evolution(
      period TEXT PRIMARY KEY, created TEXT, evaluated INTEGER, wins INTEGER, losses INTEGER,
      accuracy REAL, change_type TEXT, detail TEXT)""")
    c.execute("CREATE TABLE IF NOT EXISTS rejected_signals(horizon INTEGER,market TEXT,bucket TEXT,expires TEXT,decision TEXT,reason TEXT,entry REAL,exit REAL,confidence REAL,result TEXT,PRIMARY KEY(horizon,market,bucket))")
    c.execute("CREATE TABLE IF NOT EXISTS live_decisions(horizon INTEGER,market TEXT,bucket TEXT,decision TEXT,reason TEXT,confidence REAL,updated TEXT,PRIMARY KEY(horizon,market))")
    c.commit(); return c

def directional_quality(row,decision):
    # Familias independientes: tendencia, estructura, momentum/extremos y confluencia.
    bullish=[
        row.get("ema_trend",0)>0 or row.get("trend_pullback",0)>0,
        row.get("breakout_up",0)>0 or row.get("retest_up",0)>0 or row.get("support_bounce",0)>0,
        row.get("rsi_support",0)>0 or row.get("rsi_bull_div",0)>0 or row.get("bb_rsi_buy",0)>0,
        row.get("confluence",0)>=.125,
    ]
    bearish=[
        row.get("ema_trend",0)<0 or row.get("trend_pullback",0)<0,
        row.get("breakout_down",0)>0 or row.get("retest_down",0)>0 or row.get("resistance_bounce",0)>0,
        row.get("rsi_resistance",0)>0 or row.get("rsi_bear_div",0)>0 or row.get("bb_rsi_sell",0)>0,
        row.get("confluence",0)<=-.125,
    ]
    votes=sum(bullish if decision=="SUBE" else bearish)
    opposite=sum(bearish if decision=="SUBE" else bullish)
    return votes,opposite

def walk_forward_score(d):
    # Tres cortes temporales expansivos: cada prueba ocurre después de su entrenamiento.
    scores=[]; counts=[]
    n=len(d)
    for frac in (.55,.65,.75):
        cut=int(n*frac); end=min(n,cut+max(200,int(n*.10)))
        tr=d.iloc[:cut]; te=d.iloc[cut:end]
        if len(tr)<500 or len(te)<100: continue
        m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
        m.fit(tr[FEATURES],tr["target"].astype(int))
        p=m.predict_proba(te[FEATURES])[:,1]; y=te["target"].astype(int).to_numpy()
        take=(p>=THRESHOLD)|(p<=1-THRESHOLD)
        if take.sum()>=30:
            pred=(p>=.5).astype(int)
            scores.append(float((pred[take]==y[take]).mean())); counts.append(int(take.sum()))
    if not scores: return None,0
    return float(np.average(scores,weights=counts)),sum(counts)

def train_horizon(raw,horizon):
    d=make_features(raw,horizon,True).dropna(subset=FEATURES+["target"])
    if len(d)<500: return None
    cut=int(len(d)*.70); tr=d.iloc[:cut]; te=d.iloc[cut:]
    candidate=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    candidate.fit(tr[FEATURES],tr["target"].astype(int))
    p=candidate.predict_proba(te[FEATURES])[:,1]; y=te["target"].astype(int).to_numpy()
    take=(p>=THRESHOLD)|(p<=1-THRESHOLD); pred=(p>=.5).astype(int)
    acc=float((pred[take]==y[take]).mean()) if take.any() else None
    m=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    m.fit(d[FEATURES],d["target"].astype(int))
    return {"model":m,"features":FEATURES,"threshold":THRESHOLD,"payout":PAYOUT,"horizon":horizon,"trained_at":pd.Timestamp.now(tz="UTC").isoformat(),"validation_accuracy":acc,"validation_signals":int(take.sum())}

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
    wf_acc,wf_signals=walk_forward_score(d)
    return {"model":m,"features":FEATURES,"threshold":THRESHOLD,"payout":PAYOUT,"trained_at":pd.Timestamp.now(tz="UTC").isoformat(),"validation_accuracy":signal_acc,"validation_signals":signals,"walk_forward_accuracy":wf_acc,"walk_forward_signals":wf_signals}

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

def pair_quarantine(con,market,now,table="signals",min_sample=80):
    # Historial operativo y observaciones en sombra permanecen separados.
    # Un par bloqueado sólo se rehabilita con evidencia nueva, no por caducidad del historial.
    horizon=1 if table=="signals_1m" else 5
    since=(now-pd.Timedelta(days=7)).isoformat()
    operational=con.execute(f"SELECT result FROM {table} WHERE market=? AND bucket>=? AND result IN ('GANADA','PERDIDA')",(market,since)).fetchall()
    shadow=con.execute("SELECT result FROM rejected_signals WHERE horizon=? AND market=? AND bucket>=? AND result IN ('GANADA','PERDIDA')",(horizon,market,since)).fetchall()
    n=len(operational); wins=sum(x[0]=="GANADA" for x in operational)
    acc=wins/n if n else None
    expectancy=(wins*PAYOUT-(n-wins))/n if n else None
    key=f"quarantine_{horizon}"
    con.execute("CREATE TABLE IF NOT EXISTS quarantine_state(horizon INTEGER,market TEXT,blocked INTEGER,changed TEXT,reason TEXT,PRIMARY KEY(horizon,market))")
    previous=con.execute("SELECT blocked FROM quarantine_state WHERE horizon=? AND market=?",(horizon,market)).fetchone()
    blocked=bool(previous[0]) if previous else False
    reason="sin muestra suficiente"
    if not blocked and n>=min_sample and acc<BREAK_EVEN:
        blocked=True; reason=f"historial operativo n={n} accuracy={acc:.3f}"
    elif blocked:
        # Muestra exclusivamente posterior al bloqueo: al menos 80 señales rechazadas
        # evaluadas en los últimos 7 días y margen de 2 pp sobre equilibrio.
        sn=len(shadow); sw=sum(x[0]=="GANADA" for x in shadow)
        sa=sw/sn if sn else None
        if sn>=80 and sa>=BREAK_EVEN+.02:
            blocked=False; reason=f"recuperacion sombra n={sn} accuracy={sa:.3f}"
            # No reutilizar resultados antiguos de sombra para otra recuperación.
            con.execute("DELETE FROM rejected_signals WHERE horizon=? AND market=?",(horizon,market))
        else:
            reason=f"sombra n={sn} accuracy={sa:.3f}" if sn else "esperando señales sombra"
    if previous is None or blocked!=bool(previous[0]):
        con.execute("INSERT INTO quarantine_state VALUES(?,?,?,?,?) ON CONFLICT(horizon,market) DO UPDATE SET blocked=excluded.blocked,changed=excluded.changed,reason=excluded.reason",(horizon,market,int(blocked),now.isoformat(),reason))
        print(f"ESTADO_CUARENTENA horizon={horizon} market={market} bloqueado={blocked} motivo={reason}",flush=True)
    return not blocked,n,acc,expectancy

def record_decision(con,horizon,market,now,decision,reason,confidence,entry,raw):
    bucket=now.floor(f"{horizon}min"); expires=bucket+pd.Timedelta(minutes=horizon)
    con.execute("INSERT INTO live_decisions VALUES(?,?,?,?,?,?,?) ON CONFLICT(horizon,market) DO UPDATE SET bucket=excluded.bucket,decision=excluded.decision,reason=excluded.reason,confidence=excluded.confidence,updated=excluded.updated",(horizon,market,bucket.isoformat(),decision,reason,confidence,now.isoformat()))
    if decision=="NO OPERAR" and reason.startswith("CUARENTENA") and confidence is not None:
        # Sólo candidatos direccionales descartados por cuarentena; evaluación hipotética.
        direction=reason.split("|")[-1]
        con.execute("INSERT OR IGNORE INTO rejected_signals VALUES(?,?,?,?,?,?,?,?,?,?)",(horizon,market,bucket.isoformat(),expires.isoformat(),direction,"CUARENTENA",entry,None,confidence,"PENDIENTE"))
    for b,e,d,entry0 in con.execute("SELECT bucket,expires,decision,entry FROM rejected_signals WHERE horizon=? AND market=? AND result='PENDIENTE'",(horizon,market)).fetchall():
        exp=pd.Timestamp(e)
        if now>=exp:
            after=raw[raw["timestamp"]>=exp]
            if not after.empty:
                exit0=float(after.iloc[0].close)
                result="EMPATE" if exit0==entry0 else ("GANADA" if (d=="SUBE" and exit0>entry0) or (d=="BAJA" and exit0<entry0) else "PERDIDA")
                con.execute("UPDATE rejected_signals SET exit=?,result=? WHERE horizon=? AND market=? AND bucket=?",(exit0,result,horizon,market,b))

def hour_quality(con,market,now):
    # Sólo bloquea una franja cuando ya existe muestra forward razonable en esa misma hora UTC.
    hour=now.hour
    since=(now-pd.Timedelta(days=30)).isoformat()
    rows=con.execute("SELECT bucket,result FROM signals WHERE market=? AND bucket>=? AND result IN ('GANADA','PERDIDA')",(market,since)).fetchall()
    vals=[r for r in rows if pd.Timestamp(r[0]).hour==hour]
    n=len(vals)
    if n<40: return True,n,None
    acc=sum(r[1]=="GANADA" for r in vals)/n
    return acc>=.54,n,acc

def promote_candidate(current,candidate):
    # Evita sustituir un modelo sólo porque acaba de entrenarse.
    if current is None: return True,"primer modelo"
    ca=candidate.get("walk_forward_accuracy"); cn=candidate.get("walk_forward_signals",0)
    oa=current.get("walk_forward_accuracy",current.get("validation_accuracy"))
    if ca is None or cn<90: return False,"candidato sin muestra walk-forward suficiente"
    if oa is None: return ca>=BREAK_EVEN+.02,"comparación contra equilibrio"
    return ca>=max(BREAK_EVEN+.02,float(oa)+.005),f"walk-forward candidato={ca:.3f} actual={float(oa):.3f}"

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
    expectancy=((w*PAYOUT-l)/n) if n else None
    con.execute("INSERT OR IGNORE INTO summaries VALUES(?,?,?,?,?,?,?)",(period,now.isoformat(),len(vals),w,l,t,acc))
    con.commit()
    # Diagnóstico por contexto: qué familias aparecen más en pérdidas recientes.
    loss_rows=con.execute("SELECT context FROM signals WHERE bucket>=? AND result='PERDIDA'",(start.isoformat(),)).fetchall()
    causes={}
    for (raw_ctx,) in loss_rows:
        try:
            ctx=json.loads(raw_ctx or "{}")
            for k in STRATEGY_FIELDS:
                if abs(float(ctx.get(k,0) or 0))>0: causes[k]=causes.get(k,0)+1
        except Exception: pass
    top_causes=sorted(causes.items(),key=lambda kv:kv[1],reverse=True)[:4]
    print(f"RESUMEN_8H evaluadas={len(vals)} ganadas={w} perdidas={l} empates={t} accuracy={'NA' if acc is None else f'{acc:.3f}'} expectativa={'NA' if expectancy is None else f'{expectancy:.3f}'} causas_perdida={top_causes}",flush=True)

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
                candidate=train(raw)
                if candidate:
                    current=joblib.load(path) if path.exists() else None
                    promote,why=promote_candidate(current,candidate)
                    if promote:
                        tmp=DATA/f"model_{symbol}.tmp"; joblib.dump(candidate,tmp); tmp.replace(path)
                        meta.write_text(json.dumps({"trained_at":candidate["trained_at"],"validation_accuracy":candidate.get("validation_accuracy"),"validation_signals":candidate.get("validation_signals"),"walk_forward_accuracy":candidate.get("walk_forward_accuracy"),"walk_forward_signals":candidate.get("walk_forward_signals"),"promotion":why}))
                        print(f"{market}: modelo promovido · {why}",flush=True)
                    else:
                        # Avanza el reloj de evaluación sin reemplazar el modelo vigente.
                        meta.write_text(json.dumps({"trained_at":candidate["trained_at"],"validation_accuracy":current.get("validation_accuracy") if current else None,"validation_signals":current.get("validation_signals") if current else None,"walk_forward_accuracy":current.get("walk_forward_accuracy") if current else None,"walk_forward_signals":current.get("walk_forward_signals") if current else None,"candidate_rejected":why}))
                        print(f"{market}: candidato rechazado · {why}",flush=True)
            # Estrategia 1M independiente: objetivo, modelo e historial propios.
            path1=DATA/f"model_1m_{symbol}.joblib"; meta1=DATA/f"model_1m_{symbol}.json"
            retrain1=not path1.exists()
            if meta1.exists():
                try: retrain1=(now-pd.Timestamp(json.loads(meta1.read_text())["trained_at"])).total_seconds()>=1800
                except Exception: retrain1=True
            if retrain1:
                candidate1=train_horizon(raw,1)
                if candidate1:
                    joblib.dump(candidate1,path1)
                    meta1.write_text(json.dumps({"trained_at":candidate1["trained_at"],"validation_accuracy":candidate1.get("validation_accuracy"),"validation_signals":candidate1.get("validation_signals"),"horizon":1}))
                    print(f"{market}: modelo 1M actualizado validacion={candidate1.get('validation_accuracy')}",flush=True)
            if path1.exists():
                payload1=joblib.load(path1)
                feat1=make_features(raw,1,False).dropna(subset=FEATURES)
                if not feat1.empty:
                    row1=feat1.iloc[-1]; p1=float(payload1["model"].predict_proba(feat1[payload1["features"]].iloc[[-1]])[0,1]); th1=float(payload1.get("threshold",THRESHOLD))
                    dec1="SUBE" if p1>=th1 else ("BAJA" if p1<=1-th1 else "NO OPERAR")
                    candidate_dec1=dec1
                    reason1="modelo"
                    if dec1!="NO OPERAR":
                        votes1,opp1=directional_quality(row1,dec1)
                        if votes1<2 or opp1>=2: dec1="NO OPERAR"; reason1="confluencia"
                    pair1_ok,pair1_n,pair1_acc,pair1_exp=pair_quarantine(con,market,now,"signals_1m",80)
                    if dec1!="NO OPERAR" and not pair1_ok:
                        dec1="NO OPERAR"; reason1="CUARENTENA|"+candidate_dec1
                        print(f"CUARENTENA_1M market={market} n={pair1_n} accuracy={pair1_acc:.3f} expectativa={pair1_exp:.3f}",flush=True)
                    base1,quote1=market.split("/")
                    blocked1,_,_=news_block(now,(base1,quote1),before=15,after=30)
                    if blocked1: dec1="NO OPERAR"; reason1="noticias"
                    record_decision(con,1,market,now,dec1,reason1,max(p1,1-p1),float(raw.iloc[-1].close),raw)
                    bucket1=now.floor("1min")
                    if dec1!="NO OPERAR":
                        ctx1=json.loads(context_of(row1)); ctx1["strategy"]="1M"; ctx1=json.dumps(ctx1,separators=(",",":"))
                        con.execute("INSERT OR IGNORE INTO signals_1m VALUES(?,?,?,?,?,?,?,?,?)",(market,bucket1.isoformat(),(bucket1+pd.Timedelta(minutes=1)).isoformat(),dec1,float(raw.iloc[-1].close),None,max(p1,1-p1),"PENDIENTE",ctx1))
                    for m1,b1,e1,d1,entry1 in con.execute("SELECT market,bucket,expires,decision,entry FROM signals_1m WHERE market=? AND result='PENDIENTE'",(market,)).fetchall():
                        exp1=pd.Timestamp(e1)
                        if now>=exp1:
                            after1=raw[raw["timestamp"]>=exp1]
                            if not after1.empty:
                                exit1=float(after1.iloc[0].close); won1=(d1=="SUBE" and exit1>entry1) or (d1=="BAJA" and exit1<entry1)
                                res1="EMPATE" if exit1==entry1 else ("GANADA" if won1 else "PERDIDA")
                                con.execute("UPDATE signals_1m SET exit=?,result=? WHERE market=? AND bucket=?",(exit1,res1,m1,b1))
            if not path.exists(): continue
            payload=joblib.load(path)
            feat=make_features(raw,5,False).dropna(subset=FEATURES)
            if feat.empty: continue
            latest=feat.iloc[-1]; bucket=now.floor("5min")
            p=float(payload["model"].predict_proba(feat[payload["features"]].iloc[[-1]])[0,1])
            live_threshold,live_n,live_acc=adaptive_threshold(con,market,now)
            decision="SUBE" if p>=live_threshold else ("BAJA" if p<=1-live_threshold else "NO OPERAR")
            candidate_decision=decision
            gate_reason="modelo"
            if decision!="NO OPERAR":
                votes,opposite=directional_quality(latest,decision)
                if votes<2 or opposite>=2:
                    decision="NO OPERAR"; gate_reason=f"confluencia insuficiente votes={votes} opposite={opposite}"
            hour_ok,hour_n,hour_acc=hour_quality(con,market,now)
            if decision!="NO OPERAR" and not hour_ok:
                decision="NO OPERAR"; gate_reason=f"horario débil n={hour_n} acc={hour_acc:.3f}"
            pair_ok,pair_n,pair_acc,pair_exp=pair_quarantine(con,market,now,"signals",80)
            if decision!="NO OPERAR" and not pair_ok:
                decision="NO OPERAR"; gate_reason="CUARENTENA|"+candidate_decision
                print(f"CUARENTENA_5M market={market} n={pair_n} accuracy={pair_acc:.3f} expectativa={pair_exp:.3f}",flush=True)
            base,quote=market.split("/")
            blocked,news_reason,news_event=news_block(now,(base,quote),before=15,after=30)
            if blocked:
                decision="NO OPERAR"; gate_reason=news_reason
                print(f"NO_OPERAR_NOTICIA market={market} motivo={news_reason}",flush=True)
            record_decision(con,5,market,now,decision,gate_reason,max(p,1-p),float(raw.iloc[-1].close),raw)
            if decision!="NO OPERAR":
                ctx=json.loads(context_of(latest)); ctx["news_status"]="CLEAR"; ctx["news_source"]="Investing.com"; ctx["quality_gate"]="PASS"; ctx["pair_sample"]=live_n; ctx["pair_accuracy"]=live_acc; ctx["hour_sample"]=hour_n; ctx["hour_accuracy"]=hour_acc; ctx=json.dumps(ctx,separators=(",",":"))
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
