import asyncio
from pathlib import Path
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.deriv_market import candles
from src.binary5m import FEATURES, make_features

st.set_page_config(page_title="Binary 5M AI | Señales",page_icon="📈",layout="wide")
MARKETS={"EUR/USD":"frxEURUSD","GBP/USD":"frxGBPUSD","USD/JPY":"frxUSDJPY","AUD/USD":"frxAUDUSD","USD/CAD":"frxUSDCAD","USD/CHF":"frxUSDCHF","EUR/JPY":"frxEURJPY","GBP/JPY":"frxGBPJPY","EUR/GBP":"frxEURGBP","AUD/JPY":"frxAUDJPY"}

st.markdown("""
<style>
.block-container{max-width:1100px;padding-top:2rem}
[data-testid="stMetric"]{background:rgba(120,120,120,.08);padding:18px;border-radius:14px}
.signal-card{text-align:center;padding:28px;border:1px solid rgba(120,120,120,.25);border-radius:18px;margin:12px 0 20px}
.signal-card h1{font-size:3rem;margin:.2rem}
</style>
""",unsafe_allow_html=True)

st.title("📈 Binary 5M AI")
st.caption("Vista de señales · horizonte de 5 minutos · datos de mercado de Deriv")

with st.sidebar:
    st.header("Mercado")
    market=st.selectbox("Par",list(MARKETS))
    bars=st.slider("Velas visibles",50,200,100,10)
    st.caption("Esta vista muestra únicamente información operativa de la señal.")

symbol=MARKETS[market]

@st.cache_data(ttl=2,show_spinner=False)
def market_data(sym,count):
    return asyncio.run(candles(sym,count=count,granularity=60))

@st.fragment(run_every="2s")
def user_panel():
    try:
        raw=market_data(symbol,max(bars,250))
        last=float(raw.iloc[-1].close)
        now=pd.Timestamp.now(tz="UTC")
        next_block=now.floor("5min")+pd.Timedelta(minutes=5)
        remaining=max(0,int((next_block-now).total_seconds()))
        mm,ss=divmod(remaining,60)

        payload=None
        path=Path(f"/data/model_{symbol}.joblib")
        if path.exists():
            payload=joblib.load(path)

        decision="ESPERANDO"
        confidence=None
        if payload:
            feat=make_features(raw,5,False).dropna(subset=FEATURES)
            if not feat.empty:
                p=float(payload["model"].predict_proba(feat[payload["features"]].iloc[[-1]])[0,1])
                th=float(payload.get("threshold",.59))
                decision="SUBE" if p>=th else ("BAJA" if p<=1-th else "NO OPERAR")
                confidence=max(p,1-p)

        a,b,c=st.columns(3)
        a.metric("Par",market)
        b.metric("Precio",f"{last:.5f}" if "JPY" not in market else f"{last:.3f}")
        c.metric("Próximo bloque",f"{mm:02d}:{ss:02d}")

        icon={"SUBE":"⬆️","BAJA":"⬇️","NO OPERAR":"⏸️","ESPERANDO":"⌛"}.get(decision,"")
        st.markdown(f'<div class="signal-card"><div>SEÑAL ACTUAL</div><h1>{icon} {decision}</h1><div>{"Confianza "+format(confidence,".1%") if confidence is not None else "Preparando modelo..."}</div></div>',unsafe_allow_html=True)

        chart=raw.tail(bars)
        fig=go.Figure(data=[go.Candlestick(x=chart["timestamp"],open=chart["open"],high=chart["high"],low=chart["low"],close=chart["close"])])
        fig.update_layout(height=480,margin=dict(l=10,r=10,t=30,b=10),xaxis_rangeslider_visible=False,title=f"{market} · 1 minuto")
        st.plotly_chart(fig,width="stretch",config={"displayModeBar":False})
        st.caption("La señal es una estimación del modelo y no garantiza el resultado. Esta vista no ejecuta operaciones.")
    except Exception as e:
        st.warning(f"Mercado temporalmente no disponible: {e}")

user_panel()
