import asyncio
from pathlib import Path
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.deriv_market import candles
from src.binary5m import FEATURES, make_features, news_block, rsi

st.set_page_config(page_title="Binary 5M AI | Deriv", page_icon="📈", layout="wide")
st.title("Binary 5M AI — Deriv")
st.caption("Dashboard de investigación. Datos públicos de Deriv. No ejecuta operaciones.")

MARKETS = {
    "EUR/USD": ("frxEURUSD", "EURUSD"),
    "GBP/USD": ("frxGBPUSD", "GBPUSD"),
    "USD/JPY": ("frxUSDJPY", "USDJPY"),
    "AUD/USD": ("frxAUDUSD", "AUDUSD"),
}

with st.sidebar:
    st.header("Configuración")
    market = st.selectbox("Activo", list(MARKETS))
    bars = st.slider("Velas visibles", 50, 300, 120, 10)
    st.write("Temporalidad: **1 minuto**")
    st.write("Horizonte del modelo: **5 minutos**")
    news_file = st.file_uploader("Calendario de noticias CSV (opcional)", type="csv")
    st.caption("Pulsa Actualizar para obtener las velas más recientes.")
    refresh = st.button("🔄 Actualizar", use_container_width=True)

symbol, pair = MARKETS[market]

@st.cache_data(ttl=20, show_spinner=False)
def get_market_data(symbol, count):
    return asyncio.run(candles(symbol, count=count, granularity=60))

try:
    df = get_market_data(symbol, max(500, bars))
except Exception as e:
    st.error(f"No se pudieron cargar datos de Deriv: {e}")
    st.stop()

latest = df.iloc[-1]
prev = df.iloc[-2]
change = (latest.close / prev.close - 1) * 100

c1,c2,c3,c4 = st.columns(4)
c1.metric("Activo", market)
c2.metric("Precio", f"{latest.close:.5f}", f"{change:+.3f}%")
c3.metric("Última vela", latest.timestamp.strftime("%H:%M UTC"))
c4.metric("Horizonte", "5 min")

view=df.tail(bars)
fig=go.Figure(data=[go.Candlestick(
    x=view["timestamp"], open=view["open"], high=view["high"],
    low=view["low"], close=view["close"], name=market
)])
fig.update_layout(
    height=570, margin=dict(l=10,r=10,t=35,b=10),
    xaxis_rangeslider_visible=False,
    title=f"{market} — velas de 1 minuto",
    yaxis_title="Precio"
)
st.plotly_chart(fig, use_container_width=True)

features=make_features(df,5,False)
usable=features.dropna(subset=FEATURES)
rsi_value=float(rsi(df["close"],14).iloc[-1])
ema9=float(df["close"].ewm(span=9,adjust=False).mean().iloc[-1])
ema21=float(df["close"].ewm(span=21,adjust=False).mean().iloc[-1])
trend="ALCISTA" if ema9>ema21 else "BAJISTA"

news_path=None
if news_file is not None:
    news_df=pd.read_csv(news_file)
    news_path="/tmp/news_events.csv"
    news_df.to_csv(news_path,index=False)
currencies=(pair[:3],pair[3:6])
blocked, news_reason=news_block(latest.timestamp,news_path,currencies)

st.subheader("Análisis")
a,b,c,d=st.columns(4)
a.metric("RSI 14", f"{rsi_value:.1f}")
b.metric("Tendencia EMA 9/21", trend)
c.metric("Noticias", "BLOQUEADA" if blocked else "OK")
d.metric("Datos disponibles", f"{len(df)} velas")

model_path=Path("models/model.joblib")
st.subheader("Señal del modelo — próximos 5 minutos")
if model_path.exists() and not usable.empty:
    payload=joblib.load(model_path)
    row=usable.iloc[[-1]]
    p=float(payload["model"].predict_proba(row[payload["features"]])[0,1])
    th=float(payload["threshold"])
    if blocked:
        decision="NO OPERAR"
    elif p>=th:
        decision="SUBE"
    elif p<=1-th:
        decision="BAJA"
    else:
        decision="NO OPERAR"
    confidence=max(p,1-p)
    x,y,z=st.columns(3)
    x.metric("Decisión", decision)
    y.metric("Confianza del modelo", f"{confidence:.1%}")
    z.metric("P(SUBE) / P(BAJA)", f"{p:.1%} / {1-p:.1%}")
    st.progress(int(confidence*100))
else:
    st.warning("Todavía no hay un modelo entrenado en models/model.joblib. El gráfico y los indicadores sí funcionan; entrena el modelo antes de usar la señal.")

if blocked:
    st.error(f"Filtro de noticias: {news_reason}")
else:
    st.info(f"Filtro de noticias: {news_reason}")

with st.expander("Aviso y metodología"):
    st.write("La señal es una estimación estadística, no una certeza. Este panel no compra contratos ni envía órdenes a Deriv. Valida el modelo fuera de muestra y en demo antes de considerar cualquier decisión financiera.")
