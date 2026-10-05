import tempfile
import pandas as pd
import streamlit as st
from src.binary5m import train, signal

st.set_page_config(page_title="Binary 5M AI", layout="wide")
st.title("Binary 5M AI — laboratorio de señales")
st.caption("Investigación/backtesting. No ejecuta operaciones ni garantiza resultados.")

candles=st.file_uploader("CSV de velas de 1 minuto", type="csv")
news=st.file_uploader("Calendario de noticias (opcional)", type="csv")
payout=st.slider("Payout", .50, .95, .80, .01)
pair=st.text_input("Par", "EURUSD").upper().replace("/","")

if candles:
    raw=pd.read_csv(candles)
    st.dataframe(raw.tail(20), use_container_width=True)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        raw.to_csv(f.name,index=False); candle_path=f.name
    if st.button("Entrenar y validar"):
        try:
            train(candle_path, "models/model.joblib", payout)
            st.success("Modelo entrenado. Revisa las métricas en la terminal.")
        except Exception as e: st.error(str(e))
    if st.button("Generar señal 5m"):
        try:
            news_path=None
            if news:
                n=pd.read_csv(news)
                with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as nf:
                    n.to_csv(nf.name,index=False); news_path=nf.name
            signal(candle_path,"models/model.joblib",news_path,pair)
            st.info("La señal se imprimió en la terminal. Próxima versión: panel completo de métricas.")
        except Exception as e: st.error(str(e))
else:
    st.info("Carga un CSV con timestamp, open, high, low, close y opcionalmente volume.")
