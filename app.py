import json
from io import BytesIO
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.history_store import load as load_history, count as history_count
from src.binary5m import FEATURES, make_features, news_block, rsi

st.set_page_config(page_title="Binary 5M AI | Deriv", page_icon="📈", layout="wide")
st.title("Binary 5M AI — Deriv")
st.caption("Dashboard de investigación. Mercado almacenado localmente desde Deriv. No ejecuta operaciones.")

MARKETS={"EUR/USD":("frxEURUSD","EURUSD"),"GBP/USD":("frxGBPUSD","GBPUSD"),"USD/JPY":("frxUSDJPY","USDJPY"),"AUD/USD":("frxAUDUSD","AUDUSD"),"USD/CAD":("frxUSDCAD","USDCAD"),"USD/CHF":("frxUSDCHF","USDCHF"),"EUR/JPY":("frxEURJPY","EURJPY"),"GBP/JPY":("frxGBPJPY","GBPJPY"),"EUR/GBP":("frxEURGBP","EURGBP"),"AUD/JPY":("frxAUDJPY","AUDJPY")}

@st.cache_data(ttl=1,show_spinner=False)
def get_market_data(symbol,count):
    return load_history(symbol,count)

def fit_model(raw,payout=.80):
    data=make_features(raw,5,True).dropna(subset=FEATURES+["target"]).copy()
    if len(data)<500:
        raise ValueError("No hay suficientes velas para entrenar.")
    cut=int(len(data)*.70)
    tr,te=data.iloc[:cut],data.iloc[cut:]
    model=HistGradientBoostingClassifier(max_iter=250,learning_rate=.05,max_leaf_nodes=15,l2_regularization=1.0,random_state=42)
    model.fit(tr[FEATURES],tr["target"].astype(int))
    p=model.predict_proba(te[FEATURES])[:,1]
    y=te["target"].astype(int).to_numpy()
    threshold=max(1/(1+payout)+.03,.58)
    take=(p>=threshold)|(p<=1-threshold)
    pred=(p>=.5).astype(int)
    all_acc=float(accuracy_score(y,pred))
    trade_acc=float((pred[take]==y[take]).mean()) if take.any() else float("nan")
    pnl=float(np.where(pred[take]==y[take],payout,-1.0).sum()) if take.any() else 0.0
    return {"model":model,"features":FEATURES,"threshold":threshold,"payout":payout}, {"rows":len(data),"test":len(te),"signals":int(take.sum()),"accuracy":all_acc,"signal_accuracy":trade_acc,"pnl":pnl}

# Aplicar una recomendación pendiente ANTES de instanciar el selectbox.
if st.session_state.get("recommended_market") in MARKETS:
    st.session_state["market_selector"]=st.session_state.pop("recommended_market")

with st.sidebar:
    st.header("Navegación")
    page=st.radio("Sección",["📈 Mercado","📊 Resultados","🧠 Sistema 24/7","📝 Evolución"],index=0)
    st.divider()
    st.header("Configuración")
    market=st.selectbox("Activo",list(MARKETS),index=0,key="market_selector",help="Al abrir Mercado se selecciona automáticamente el mejor par operable; puedes cambiarlo manualmente.")
    bars=st.slider("Velas visibles",50,300,120,10)
    st.write("Temporalidad: **1 minuto**")
    st.write("Horizonte: **5 minutos**")
    news_file=st.file_uploader("Calendario de noticias CSV (opcional)",type="csv")
    st.caption("El mercado se actualiza en vivo. La alerta del modelo solo cambia una vez cada 5 minutos.")
    if st.button("🔄 Actualizar mercado",width="stretch"):
        get_market_data.clear()
    st.divider()
    st.subheader("Entrenamiento automático")
    train_count=st.select_slider("Velas para entrenar",options=[1000,2000,3000,5000,10000,25000,50000,100000],value=100000)
    st.caption("El modelo se entrena automáticamente al abrir o al cambiar de activo.")

symbol,pair=MARKETS[market]

@st.cache_data(ttl=60,show_spinner=False)
def rank_markets(count):
    rows=[]
    for name,(sym,_) in MARKETS.items():
        try:
            raw=load_history(sym,count)
            if raw.empty: continue
            model_data,metrics=fit_model(raw)
            feat=make_features(raw,5,False).dropna(subset=FEATURES)
            if feat.empty:
                continue
            p=float(model_data["model"].predict_proba(feat[FEATURES].iloc[[-1]])[0,1])
            th=float(model_data["threshold"])
            decision="SUBE" if p>=th else ("BAJA" if p<=1-th else "NO OPERAR")
            confidence=max(p,1-p)
            conf=float(feat.iloc[-1].get("confluence",0.0))
            hist_acc=metrics["signal_accuracy"]
            hist_score=0.5 if np.isnan(hist_acc) else float(hist_acc)
            score=(confidence*.50)+(hist_score*.35)+(min(abs(conf),1.0)*.15)
            rows.append({"Par":name,"Señal":decision,"Confianza":confidence,"Confluencia":conf,"Accuracy prueba":hist_acc,"Score":score})
        except Exception:
            continue
    return pd.DataFrame(rows).sort_values("Score",ascending=False) if rows else pd.DataFrame()

if page=="📈 Mercado":
  ranking=rank_markets(min(train_count,3000))
  operables=ranking[ranking["Señal"]!="NO OPERAR"] if not ranking.empty else ranking
  if not operables.empty:
      best=operables.iloc[0]
      best_market=str(best["Par"])
      # El selectbox ya existe en esta ejecución; no mutamos su key aquí.
      # Guardamos la recomendación y la aplicamos de forma segura en el siguiente rerun.
      if not st.session_state.get("auto_best_selected"):
          st.session_state["auto_best_selected"]=True
          st.session_state["recommended_market"]=best_market
  with st.expander("🏆 Selección automática",expanded=False):
      ranking=rank_markets(min(train_count,3000))
      operables=ranking[ranking["Señal"]!="NO OPERAR"] if not ranking.empty else ranking
      if operables.empty:
          st.warning("NO OPERAR: ninguno de los pares supera ahora el umbral del modelo.")
      else:
          best=operables.iloc[0]
          b1,b2,b3=st.columns(3)
          b1.metric("Mejor par",best["Par"])
          b2.metric("Señal",best["Señal"])
          b3.metric("Confianza",f"{best['Confianza']:.1%}")
          show=ranking.copy()
          show["Confianza"]=show["Confianza"].map(lambda v:f"{v:.1%}")
          show["Accuracy prueba"]=show["Accuracy prueba"].map(lambda v:"—" if pd.isna(v) else f"{v:.1%}")
          show["Confluencia"]=show["Confluencia"].map(lambda v:f"{v:+.2f}")
          st.dataframe(show[["Par","Señal","Confianza","Confluencia","Accuracy prueba"]],width="stretch",hide_index=True)
          st.caption("Ranking orientativo para el próximo horizonte de 5 minutos. Compara confianza, validación histórica y confluencia; no garantiza el resultado.")
  
market=st.session_state.get("market_selector",market)
symbol,pair=MARKETS[market]

if page in ("📈 Mercado","🧠 Sistema 24/7"):
    auto_key=f"trained_{market}_{train_count}"
    if not st.session_state.get(auto_key):
        with st.spinner(f"Entrenando automáticamente {market}..."):
            try:
                hist=load_history(symbol,train_count)
                if len(hist)<500: raise ValueError(f"Histórico local insuficiente: {len(hist)} velas")
                payload,metrics=fit_model(hist)
                st.session_state["model_payload"]=payload
                st.session_state["model_market"]=market
                st.session_state["model_metrics"]=metrics
                st.session_state["trained_at"]=pd.Timestamp.now(tz="UTC")
                st.session_state[auto_key]=True
                st.success("Modelo listo. Alertas automáticas activadas.")
            except Exception as e:
                st.error(f"No se pudo entrenar automáticamente: {e}")

def persistent_learning_panel():
    db_path=Path("/data/learning.db")
    st.subheader("📚 Aprendizaje 24/7")
    if not db_path.exists():
        st.info("El worker 24/7 todavía no ha creado datos persistentes.")
        return
    try:
        import sqlite3
        con=sqlite3.connect(f"file:{db_path}?mode=ro",uri=True,timeout=2)
        hist=pd.read_sql_query("SELECT market,bucket,expires,decision,entry,exit,confidence,result,context FROM signals ORDER BY bucket DESC",con)
        con.close()
    except Exception as e:
        st.warning(f"No se pudo leer el aprendizaje persistente: {e}")
        return
    if hist.empty:
        st.info("El worker está activo, pero todavía no hay señales registradas.")
        return

    evaluated=hist[hist["result"].isin(["GANADA","PERDIDA","EMPATE"])].copy()
    wins=int((evaluated["result"]=="GANADA").sum())
    losses=int((evaluated["result"]=="PERDIDA").sum())
    ties=int((evaluated["result"]=="EMPATE").sum())
    decided=wins+losses
    accuracy=(wins/decided) if decided else float("nan")
    pending=int((hist["result"]=="PENDIENTE").sum())

    p1,p2,p3,p4,p5=st.columns(5)
    p1.metric("Evaluadas",len(evaluated))
    p2.metric("Ganadas",wins)
    p3.metric("Perdidas",losses)
    p4.metric("Accuracy","—" if np.isnan(accuracy) else f"{accuracy:.1%}")
    p5.metric("Pendientes",pending)

    recent=evaluated.head(30).copy()
    recent_decided=recent[recent["result"].isin(["GANADA","PERDIDA"])]
    rw=int((recent_decided["result"]=="GANADA").sum()); rn=len(recent_decided)
    recent_acc=(rw/rn) if rn else float("nan")
    if decided<30:
        status="⚪ APRENDIENDO"; status_msg="Todavía hay poca muestra para juzgar el sistema."
    elif accuracy>=.62:
        status="🟢 FUERTE"; status_msg="El histórico evaluado está por encima del nivel de referencia."
    elif accuracy>=.56:
        status="🟡 VIGILAR"; status_msg="Está sobre el punto de equilibrio aproximado, pero necesita más evidencia."
    else:
        status="🔴 DÉBIL"; status_msg="El rendimiento evaluado está por debajo del nivel de referencia."
    st.markdown(f"### Estado general: {status}")
    st.caption(status_msg)
    v1,v2,v3=st.columns(3)
    v1.metric("Acierto total","—" if np.isnan(accuracy) else f"{accuracy:.1%}")
    v2.metric("Últimas 30","—" if np.isnan(recent_acc) else f"{recent_acc:.1%}",None if np.isnan(recent_acc) or np.isnan(accuracy) else f"{(recent_acc-accuracy)*100:+.1f} pp vs total")
    v3.metric("Muestra útil",decided)
    if decided:
        left,right=st.columns([1,1.7])
        with left:
            donut=go.Figure(go.Pie(labels=["Ganadas","Perdidas"],values=[wins,losses],hole=.68,textinfo="label+percent",marker=dict(colors=["#A8E6CF","#FFAAA5"],line=dict(color="#20242c",width=2)),hovertemplate="%{label}: %{value}<extra></extra>"))
            donut.update_layout(title="Balance de resultados",height=330,margin=dict(l=20,r=20,t=55,b=20),showlegend=False,annotations=[dict(text=f"{accuracy:.1%}",x=.5,y=.5,font_size=25,showarrow=False)])
            st.plotly_chart(donut,width="stretch",key="results-donut")
        with right:
            trend=evaluated[evaluated["result"].isin(["GANADA","PERDIDA"])].copy()
            trend["bucket"]=pd.to_datetime(trend["bucket"],utc=True,errors="coerce")
            trend=trend.sort_values("bucket").tail(80)
            trend["win"]=(trend["result"]=="GANADA").astype(int)
            trend["Acierto móvil"]=trend["win"].rolling(15,min_periods=5).mean()*100
            tf=go.Figure()
            trend["Acierto 30"]=trend["win"].rolling(30,min_periods=8).mean()*100
            tf.add_trace(go.Scatter(x=trend["bucket"],y=trend["Acierto móvil"],mode="lines+markers",name="Media 15",line=dict(width=3,color="#B8B5FF"),marker=dict(color="#B8B5FF")))
            tf.add_trace(go.Scatter(x=trend["bucket"],y=trend["Acierto 30"],mode="lines",name="Media 30",line=dict(width=3,color="#FFD3B6")))
            tf.add_hline(y=accuracy*100,line_dash="dot",line_color="#A8D8EA",annotation_text="Promedio total")
            tf.add_hline(y=55.6,line_dash="dash",line_color="#FFAAA5",annotation_text="Referencia 55.6%")
            tf.update_layout(title="Tendencia reciente",height=330,margin=dict(l=20,r=20,t=55,b=20),yaxis=dict(title="Acierto %",range=[0,100]),xaxis_title=None,hovermode="x unified")
            st.plotly_chart(tf,width="stretch",key="results-trend")

    rows=[]
    for name,g in evaluated.groupby("market"):
        w=int((g["result"]=="GANADA").sum())
        l=int((g["result"]=="PERDIDA").sum())
        t=int((g["result"]=="EMPATE").sum())
        n=w+l
        rows.append({"Par":name,"Evaluadas":len(g),"Ganadas":w,"Perdidas":l,"Empates":t,"Accuracy":(w/n if n else np.nan)})
    if rows:
        by_pair=pd.DataFrame(rows).sort_values(["Accuracy","Evaluadas"],ascending=[False,False])
        st.markdown("### Rendimiento por par")
        visual=by_pair.dropna(subset=["Accuracy"]).sort_values("Accuracy",ascending=True)
        if not visual.empty:
            pf=go.Figure(go.Bar(x=visual["Accuracy"]*100,y=visual["Par"],orientation="h",text=visual["Accuracy"].map(lambda v:f"{v:.1%}"),textposition="outside",marker=dict(color=["#FFB7B2","#FFDAC1","#E2F0CB","#B5EAD7","#C7CEEA","#D5AAFF","#A8D8EA","#F6C1C7","#B8E0D2","#F3D1F4"][:len(visual)]),hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
            pf.add_vline(x=55.6,line_dash="dash",line_color="#FFAAA5",annotation_text="Referencia 55.6%")
            pf.update_layout(title="Comparación de pares",height=max(360,55*len(visual)),margin=dict(l=20,r=70,t=55,b=35),xaxis=dict(title="Acierto %",range=[0,100]),yaxis_title=None,showlegend=False)
            st.plotly_chart(pf,width="stretch",key="pair-performance")
        by_pair["Indicador"]=by_pair.apply(lambda r:"⚪ Poca muestra" if (r["Ganadas"]+r["Perdidas"])<30 else ("🟢 Fuerte" if r["Accuracy"]>=.62 else ("🟡 Vigilar" if r["Accuracy"]>=.56 else "🔴 Débil")),axis=1)
        by_pair["Accuracy"]=by_pair["Accuracy"].map(lambda v:"—" if pd.isna(v) else f"{v:.1%}")
        st.dataframe(by_pair[["Par","Indicador","Evaluadas","Ganadas","Perdidas","Accuracy"]],width="stretch",hide_index=True)

    # Evolución acumulada y exportación para análisis personal.
    decided_curve=evaluated[evaluated["result"].isin(["GANADA","PERDIDA"])].copy()
    if not decided_curve.empty:
        decided_curve["bucket"]=pd.to_datetime(decided_curve["bucket"],utc=True,errors="coerce")
        decided_curve=decided_curve.sort_values("bucket")
        decided_curve["Balance acumulado"]=np.where(decided_curve["result"]=="GANADA",.80,-1.0).cumsum()
        cf=go.Figure(go.Scatter(x=decided_curve["bucket"],y=decided_curve["Balance acumulado"],mode="lines",fill="tozeroy",name="Balance",line=dict(width=3,color="#B5EAD7"),fillcolor="rgba(181,234,215,0.18)"))
        cf.add_hline(y=0,line_dash="dash",line_color="#FFAAA5")
        cf.update_layout(title="Evolución acumulada (unidades teóricas)",height=340,margin=dict(l=20,r=20,t=55,b=25),xaxis_title=None,yaxis_title="Unidades",hovermode="x unified")
        st.plotly_chart(cf,width="stretch",key="cumulative-results")

    export_hist=hist.copy()
    export_summary=pd.DataFrame([{"Evaluadas":len(evaluated),"Ganadas":wins,"Perdidas":losses,"Empates":ties,"Pendientes":pending,"Accuracy":None if np.isnan(accuracy) else accuracy}])
    notes=pd.DataFrame(columns=["Fecha","Par","Señal","Anotación personal","Qué observé","Seguimiento"])
    output=BytesIO()
    with pd.ExcelWriter(output,engine="openpyxl") as writer:
        export_summary.to_excel(writer,index=False,sheet_name="Resumen")
        if rows:
            pd.DataFrame(rows).to_excel(writer,index=False,sheet_name="Por par")
        export_hist.to_excel(writer,index=False,sheet_name="Señales")
        notes.to_excel(writer,index=False,sheet_name="Notas")
    st.download_button("📥 Descargar resultados en Excel",data=output.getvalue(),file_name=f"binary_5m_resultados_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M')}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",width="stretch")
    st.caption("El Excel es una copia para tus anotaciones; descargarlo no modifica ni reinicia el historial guardado en la aplicación.")

    last=hist.head(20).copy()
    last["Confianza"]=pd.to_numeric(last["confidence"],errors="coerce").map(lambda v:"—" if pd.isna(v) else f"{v:.1%}")
    last["bucket"]=pd.to_datetime(last["bucket"],utc=True,errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    st.caption("Últimas 20 señales guardadas por el worker 24/7")
    st.dataframe(last[["bucket","market","decision","Confianza","result"]].rename(columns={"bucket":"Hora UTC","market":"Par","decision":"Señal","result":"Resultado"}),width="stretch",hide_index=True)
    newest=pd.to_datetime(hist["bucket"],utc=True,errors="coerce").max()
    if pd.notna(newest):
        st.caption(f"Último ciclo registrado: {newest.strftime('%Y-%m-%d %H:%M UTC')}. Los resultados persistentes continúan aunque cierres esta página.")
    try:
        summaries=pd.read_sql_query("SELECT period,evaluated,wins,losses,ties,accuracy FROM summaries ORDER BY period DESC LIMIT 12",sqlite3.connect(f"file:{db_path}?mode=ro",uri=True))
        if not summaries.empty:
            st.markdown("**Resumen por bloques de 8 horas**")
            summaries["accuracy"]=summaries["accuracy"].map(lambda v:"—" if pd.isna(v) else f"{v:.1%}")
            st.dataframe(summaries.rename(columns={"period":"Periodo UTC","evaluated":"Evaluadas","wins":"Ganadas","losses":"Perdidas","ties":"Empates","accuracy":"Accuracy"}),width="stretch",hide_index=True)
    except Exception:
        pass
    decided=hist[hist["result"].isin(["GANADA","PERDIDA"])].copy()
    if not decided.empty:
        decided["Hora UTC"]=pd.to_datetime(decided["bucket"],utc=True,errors="coerce").dt.hour
        perf=[]
        for (pair,hour),g in decided.groupby(["market","Hora UTC"]):
            w=int((g["result"]=="GANADA").sum()); n=len(g)
            perf.append({"Par":pair,"Hora UTC":int(hour),"Muestra":n,"Ganadas":w,"Perdidas":n-w,"Accuracy":w/n})
        perf=pd.DataFrame(perf).sort_values(["Accuracy","Muestra"],ascending=[False,False])
        perf["Estado"]=perf.apply(lambda r:"Muestra pequeña" if r["Muestra"]<30 else ("Evitar/elevar umbral" if r["Accuracy"]<.56 else "Válido"),axis=1)
        perf["Accuracy"]=perf["Accuracy"].map(lambda v:f"{v:.1%}")
        with st.expander("🕒 Rendimiento por par y hora"):
            st.dataframe(perf.head(40),width="stretch",hide_index=True)
            st.caption("El filtro adaptativo solo usa rendimiento forward con al menos 30 resultados; una pérdida aislada no modifica el modelo.")

    model_rows=[]
    for pair,(sym,_) in MARKETS.items():
        meta_path=Path(f"/data/model_{sym}.json")
        if meta_path.exists():
            try:
                md=json.loads(meta_path.read_text())
                va=md.get("validation_accuracy"); vs=md.get("validation_signals",0)
                model_rows.append({"Par":pair,"Último entrenamiento":md.get("trained_at","—"),"Señales validación":vs,"Accuracy walk-forward":("—" if va is None else f"{va:.1%}")})
            except Exception:
                pass
    if model_rows:
        with st.expander("🧪 Validación walk-forward"):
            st.dataframe(pd.DataFrame(model_rows),width="stretch",hide_index=True)
            st.caption("La validación usa el 30% cronológicamente posterior antes de ajustar el modelo de producción con todo el histórico.")

    failures=hist[hist["result"]=="PERDIDA"].head(10)
    if not failures.empty:
        with st.expander("🔎 Diagnóstico de los últimos fallos"):
            st.caption("Guarda el contexto técnico de cada pérdida para detectar patrones repetidos. No cambia el modelo por una sola pérdida; esos datos se usan para comparar y corregir con suficiente muestra.")
            for _,r in failures.iterrows():
                try:
                    ctx=json.loads(r.get("context") or "{}")
                    active=[k for k,v in ctx.items() if k!="confluence" and abs(float(v))>0]
                    st.write(f"{r['market']} · {r['decision']} · confianza {float(r['confidence']):.1%} · confluencia {float(ctx.get('confluence',0)):+.2f} · condiciones: {', '.join(active[:6]) or 'sin condición fuerte'}")
                except Exception:
                    pass

if page=="📊 Resultados":
    persistent_learning_panel()

if page=="📝 Evolución":
    st.subheader("📝 Bitácora de evolución")
    st.caption("Qué evaluó el sistema cada hora y qué decisión tomó. No muestra cambios ficticios: si no hubo una modificación real, lo indica.")
    db_path=Path("/data/learning.db")
    if db_path.exists():
        try:
            import sqlite3
            con=sqlite3.connect(f"file:{db_path}?mode=ro",uri=True,timeout=2)
            evo=pd.read_sql_query("SELECT period,created,evaluated,wins,losses,accuracy,change_type,detail FROM evolution ORDER BY period DESC LIMIT 168",con)
            con.close()
            if evo.empty:
                st.info("La bitácora comenzará a llenarse al cerrar el próximo bloque horario.")
            else:
                evo["accuracy"]=evo["accuracy"].map(lambda v:"—" if pd.isna(v) else f"{v:.1%}")
                st.dataframe(evo.rename(columns={"period":"Hora UTC","evaluated":"Evaluadas","wins":"Ganadas","losses":"Perdidas","accuracy":"Accuracy","change_type":"Tipo","detail":"Qué hizo / por qué"}),width="stretch",hide_index=True)
        except Exception as e:
            st.info(f"La bitácora se está inicializando: {e}")
    else:
        st.info("Esperando que el worker cree la bitácora persistente.")


@st.fragment(run_every="2s")
def live_panel():
    try:
        get_market_data.clear()
        df=get_market_data(symbol,max(500,bars))
    except Exception as e:
        st.error(f"No se pudieron cargar datos del almacén local: {e}")
        return

    latest,prev=df.iloc[-1],df.iloc[-2]
    change=(latest.close/prev.close-1)*100
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Activo",market)
    c2.metric("Precio EN VIVO",f"{latest.close:.5f}",f"{change:+.3f}%")
    c3.metric("Última vela",latest.timestamp.strftime("%H:%M UTC"))
    c4.metric("Estado","🟢 EN VIVO")

    view=df.tail(bars)
    fig=go.Figure(data=[go.Candlestick(x=view.timestamp,open=view.open,high=view.high,low=view.low,close=view.close,name=market)])
    fig.update_layout(height=570,margin=dict(l=10,r=10,t=35,b=10),xaxis_rangeslider_visible=False,title=f"{market} — velas de 1 minuto",yaxis_title="Precio",uirevision=market)
    st.plotly_chart(fig,use_container_width=True,key=f"live-chart-{market}")

    features=make_features(df,5,False)
    usable=features.dropna(subset=FEATURES)
    rsi_value=float(rsi(df.close,14).iloc[-1])
    ema9=float(df.close.ewm(span=9,adjust=False).mean().iloc[-1])
    ema21=float(df.close.ewm(span=21,adjust=False).mean().iloc[-1])
    trend="ALCISTA" if ema9>ema21 else "BAJISTA"

    news_path=None
    if news_file is not None:
        news_df=pd.read_csv(news_file)
        news_path="/tmp/news_events.csv"
        news_df.to_csv(news_path,index=False)
    blocked,news_reason=news_block(latest.timestamp,news_path,(pair[:3],pair[3:6]))

    st.subheader("Análisis en vivo")
    a,b,c,d=st.columns(4)
    a.metric("RSI 14",f"{rsi_value:.1f}")
    b.metric("Tendencia EMA 9/21",trend)
    c.metric("Noticias","BLOQUEADA" if blocked else "OK")
    d.metric("Datos disponibles",f"{len(df)} velas")

    payload=st.session_state.get("model_payload") if st.session_state.get("model_market")==market else None
    persistent_model=Path(f"/data/model_{symbol}.joblib")
    if persistent_model.exists():
        try:
            payload=joblib.load(persistent_model)
        except Exception:
            pass
    if payload is None and Path("models/model.joblib").exists():
        payload=joblib.load("models/model.joblib")

    now=pd.Timestamp.now(tz="UTC")
    alert_bucket=now.floor("5min")
    next_alert=alert_bucket+pd.Timedelta(minutes=5)
    remaining=max(0,int((next_alert-now).total_seconds()))

    # Evalúa alertas anteriores exactamente con datos posteriores, sin look-ahead.
    history_key=f"live_history_{market}"
    history=st.session_state.setdefault(history_key,[])
    for rec in history:
        if rec["result"]=="PENDIENTE" and now >= rec["expires"]:
            after=df[df["timestamp"] >= rec["expires"]]
            if not after.empty:
                exit_price=float(after.iloc[0]["close"])
                rec["exit_price"]=exit_price
                if exit_price == rec["entry_price"]:
                    rec["result"]="EMPATE"
                else:
                    won=(rec["decision"]=="SUBE" and exit_price>rec["entry_price"]) or (rec["decision"]=="BAJA" and exit_price<rec["entry_price"])
                    rec["result"]="GANADA" if won else "PERDIDA"

    # Reentrena cada 30 minutos con las velas más recientes.
    trained_at=st.session_state.get("trained_at")
    if trained_at is not None and (now-trained_at).total_seconds() >= 1800:
        try:
            fresh=load_history(symbol,train_count)
            if len(fresh)<500: raise ValueError("Histórico local insuficiente")
            new_payload,new_metrics=fit_model(fresh)
            st.session_state["model_payload"]=new_payload
            st.session_state["model_metrics"]=new_metrics
            st.session_state["trained_at"]=now
            payload=new_payload
        except Exception:
            pass

    st.subheader("Alerta de 5 minutos")
    if payload is not None and not usable.empty:
        alert_key=f"alert_{market}"
        saved=st.session_state.get(alert_key)
        signal_bucket=next_alert if remaining <= 10 else alert_bucket
        if saved is None or saved.get("bucket") != signal_bucket:
            p=float(payload["model"].predict_proba(usable[payload["features"]].iloc[[-1]])[0,1])
            th=float(payload["threshold"])
            decision="NO OPERAR" if blocked else ("SUBE" if p>=th else ("BAJA" if p<=1-th else "NO OPERAR"))
            st.session_state[alert_key]={"bucket":signal_bucket,"decision":decision,"p":p,"confidence":max(p,1-p)}
            if decision in ("SUBE","BAJA") and not any(r["bucket"]==signal_bucket for r in history):
                history.append({"bucket":signal_bucket,"expires":signal_bucket+pd.Timedelta(minutes=5),"decision":decision,"entry_price":float(latest.close),"exit_price":None,"confidence":max(p,1-p),"result":"PENDIENTE"})
                if len(history)>200:
                    del history[:-200]
        saved=st.session_state[alert_key]
        x,y,z,w=st.columns(4)
        x.metric("Alerta",saved["decision"])
        y.metric("Confianza",f"{saved['confidence']:.1%}")
        z.metric("P(SUBE) / P(BAJA)",f"{saved['p']:.1%} / {1-saved['p']:.1%}")
        w.metric("Próxima alerta",f"{remaining//60:02d}:{remaining%60:02d}")
        if saved["bucket"] > alert_bucket:
            st.success(f"⚡ ALERTA ANTICIPADA: señal para {saved['bucket'].strftime('%H:%M')}–{(saved['bucket']+pd.Timedelta(minutes=5)).strftime('%H:%M')} UTC. Faltan {remaining} s para iniciar.")
        else:
            st.caption(f"Señal fija para {saved['bucket'].strftime('%H:%M')}–{(saved['bucket']+pd.Timedelta(minutes=5)).strftime('%H:%M')} UTC.")
    else:
        st.warning("El modelo automático todavía se está preparando. La alerta aparecerá en cuanto esté listo.")

    if page!="📈 Mercado":
        st.subheader("Aprendizaje en vivo")
        completed=[r for r in history if r["result"] in ("GANADA","PERDIDA")]
        wins=sum(r["result"]=="GANADA" for r in completed)
        losses=sum(r["result"]=="PERDIDA" for r in completed)
        rate=(wins/len(completed)*100) if completed else None
        l1,l2,l3,l4=st.columns(4)
        l1.metric("Alertas evaluadas",len(completed))
        l2.metric("Ganadas",wins)
        l3.metric("Perdidas",losses)
        l4.metric("Acierto en vivo","—" if rate is None else f"{rate:.1f}%")
        st.caption("El modelo registra cada señal, comprueba el resultado 5 minutos después y se reentrena cada 30 minutos con las velas más recientes. Más reentrenamiento no garantiza mayor precisión.")
        if history:
            table=pd.DataFrame(history[-20:]).copy()
            table["hora"]=table["bucket"].apply(lambda x:x.strftime("%H:%M"))
            table["confianza"]=table["confidence"].apply(lambda x:f"{x:.1%}")
            st.dataframe(table[["hora","decision","confianza","entry_price","exit_price","result"]].iloc[::-1],width="stretch",hide_index=True)
    
    if blocked:
        st.error(f"Filtro de noticias: {news_reason}")
    else:
        st.info(f"Filtro de noticias: {news_reason}")

if page=="📈 Mercado":
    live_panel()

if page=="🧠 Sistema 24/7":
    m=st.session_state.get("model_metrics")
    if m and st.session_state.get("model_market")==market:
        st.subheader("Validación fuera de muestra")
        q1,q2,q3,q4=st.columns(4)
        q1.metric("Accuracy total",f"{m['accuracy']:.1%}")
        q2.metric("Accuracy señales",("—" if np.isnan(m["signal_accuracy"]) else f"{m['signal_accuracy']:.1%}"))
        q3.metric("Señales de prueba",m["signals"])
        q4.metric("Resultado teórico",f"{m['pnl']:.2f} u")
        st.caption(f"Entrenamiento/prueba separados en orden temporal 70/30. {m['rows']} observaciones útiles; {m['test']} en prueba. Resultado teórico usa payout 80% y no garantiza rendimiento futuro.")

with st.expander("Aviso y metodología"):
    st.write("La señal es una estimación estadística, no una certeza. El panel no compra contratos ni envía órdenes. Un resultado positivo en la prueba histórica no garantiza beneficios futuros; valida también en demo y con muestras más largas.")
