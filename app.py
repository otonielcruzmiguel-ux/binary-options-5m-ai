import asyncio
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

from src.deriv_market import candles
from src.binary5m import FEATURES, make_features, news_block, rsi

st.set_page_config(page_title="Binary 5M AI | Deriv", page_icon="📈", layout="wide")
st.title("Binary 5M AI — Deriv")
st.caption("Dashboard de investigación. Datos públicos de Deriv. No ejecuta operaciones.")
st_autorefresh(interval=1000, limit=None, key="market-live-refresh")

MARKETS={"EUR/USD":("frxEURUSD","EURUSD"),"GBP/USD":("frxGBPUSD","GBPUSD"),"USD/JPY":("frxUSDJPY","USDJPY"),"AUD/USD":("frxAUDUSD","AUDUSD")}

@st.cache_data(ttl=1,show_spinner=False)
def get_market_data(symbol,count):
    return asyncio.run(candles(symbol,count=count,granularity=60))

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

with st.sidebar:
    st.header("Configuración")
    market=st.selectbox("Activo",list(MARKETS))
    bars=st.slider("Velas visibles",50,300,120,10)
    st.write("Temporalidad: **1 minuto**")
    st.write("Horizonte: **5 minutos**")
    news_file=st.file_uploader("Calendario de noticias CSV (opcional)",type="csv")
    if st.button("🔄 Actualizar",use_container_width=True):
        get_market_data.clear()
    st.divider()
    st.subheader("Entrenamiento")
    train_count=st.select_slider("Velas para entrenar",options=[1000,2000,3000,5000],value=5000)
    train_button=st.button("🧠 Entrenar modelo",use_container_width=True,type="primary")

symbol,pair=MARKETS[market]

if train_button:
    with st.spinner(f"Descargando {train_count} velas y entrenando {market}..."):
        try:
            hist=asyncio.run(candles(symbol,count=train_count,granularity=60))
            payload,metrics=fit_model(hist)
            st.session_state["model_payload"]=payload
            st.session_state["model_market"]=market
            st.session_state["model_metrics"]=metrics
            st.success("Modelo entrenado y validado cronológicamente.")
        except Exception as e:
            st.error(f"No se pudo entrenar: {e}")

try:
    df=get_market_data(symbol,max(500,bars))
except Exception as e:
    st.error(f"No se pudieron cargar datos de Deriv: {e}")
    st.stop()

latest,prev=df.iloc[-1],df.iloc[-2]
change=(latest.close/prev.close-1)*100
c1,c2,c3,c4=st.columns(4)
c1.metric("Activo",market); c2.metric("Precio",f"{latest.close:.5f}",f"{change:+.3f}%")
c3.metric("Última vela",latest.timestamp.strftime("%H:%M:%S UTC")); c4.metric("Estado","🟢 EN VIVO")

view=df.tail(bars)
fig=go.Figure(data=[go.Candlestick(x=view.timestamp,open=view.open,high=view.high,low=view.low,close=view.close,name=market)])
fig.update_layout(height=570,margin=dict(l=10,r=10,t=35,b=10),xaxis_rangeslider_visible=False,title=f"{market} — velas de 1 minuto",yaxis_title="Precio")
st.plotly_chart(fig,use_container_width=True)

features=make_features(df,5,False)
usable=features.dropna(subset=FEATURES)
rsi_value=float(rsi(df.close,14).iloc[-1])
ema9=float(df.close.ewm(span=9,adjust=False).mean().iloc[-1]); ema21=float(df.close.ewm(span=21,adjust=False).mean().iloc[-1])
trend="ALCISTA" if ema9>ema21 else "BAJISTA"
news_path=None
if news_file is not None:
    news_df=pd.read_csv(news_file); news_path="/tmp/news_events.csv"; news_df.to_csv(news_path,index=False)
blocked,news_reason=news_block(latest.timestamp,news_path,(pair[:3],pair[3:6]))

st.subheader("Análisis")
a,b,c,d=st.columns(4)
a.metric("RSI 14",f"{rsi_value:.1f}"); b.metric("Tendencia EMA 9/21",trend)
c.metric("Noticias","BLOQUEADA" if blocked else "OK"); d.metric("Datos disponibles",f"{len(df)} velas")

payload=None
if st.session_state.get("model_market")==market:
    payload=st.session_state.get("model_payload")
elif Path("models/model.joblib").exists():
    payload=joblib.load("models/model.joblib")

st.subheader("Señal del modelo — próximos 5 minutos")
if payload is not None and not usable.empty:
    p=float(payload["model"].predict_proba(usable[payload["features"]].iloc[[-1]])[0,1])
    th=float(payload["threshold"])
    decision="NO OPERAR" if blocked else ("SUBE" if p>=th else ("BAJA" if p<=1-th else "NO OPERAR"))
    confidence=max(p,1-p)
    x,y,z=st.columns(3)
    x.metric("Decisión",decision); y.metric("Confianza del modelo",f"{confidence:.1%}"); z.metric("P(SUBE) / P(BAJA)",f"{p:.1%} / {1-p:.1%}")
    st.progress(int(confidence*100))
else:
    st.warning("Pulsa «Entrenar modelo» en la barra lateral para crear un modelo con datos históricos de este activo.")

m=st.session_state.get("model_metrics")
if m and st.session_state.get("model_market")==market:
    st.subheader("Validación fuera de muestra")
    q1,q2,q3,q4=st.columns(4)
    q1.metric("Accuracy total",f"{m['accuracy']:.1%}")
    q2.metric("Accuracy señales",("—" if np.isnan(m["signal_accuracy"]) else f"{m['signal_accuracy']:.1%}"))
    q3.metric("Señales de prueba",m["signals"])
    q4.metric("Resultado teórico",f"{m['pnl']:.2f} u")
    st.caption(f"Entrenamiento/prueba separados en orden temporal 70/30. {m['rows']} observaciones útiles; {m['test']} en prueba. Resultado teórico usa payout 80% y no garantiza rendimiento futuro.")

st.error(f"Filtro de noticias: {news_reason}") if blocked else st.info(f"Filtro de noticias: {news_reason}")
with st.expander("Aviso y metodología"):
    st.write("La señal es una estimación estadística, no una certeza. El panel no compra contratos ni envía órdenes. Un resultado positivo en la prueba histórica no garantiza beneficios futuros; valida también en demo y con muestras más largas.")
