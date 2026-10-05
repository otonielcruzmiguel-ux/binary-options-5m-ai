# Binary Options 5M AI

Laboratorio educativo en Python para estudiar si un activo estará **arriba o abajo dentro de 5 minutos** usando velas de 1 minuto.

> **Importante:** no es asesoría financiera, no garantiza ganancias y no ejecuta órdenes. Las opciones binarias son de alto riesgo. Valida primero con datos fuera de muestra y cuenta demo.

## Qué incluye

- Objetivo de predicción a 5 minutos.
- Indicadores/features: retornos, EMA 9/21, RSI 14, ATR 14, Bollinger, rango, cuerpo de vela, volatilidad y hora.
- HistGradientBoostingClassifier.
- División cronológica 70/30 para reducir fuga de información.
- Umbral de **NO OPERAR** cuando la confianza es insuficiente.
- Backtest simple considerando payout.
- Filtro de noticias de alto impacto a ±15 minutos mediante un calendario CSV.
- Interfaz inicial con Streamlit.
- No usa martingala y no se conecta a un broker.

## Formato de velas

CSV de 1 minuto con columnas: timestamp, open, high, low, close, volume.
Los timestamps se interpretan en UTC. Para entrenar se requieren al menos 500 filas; para una prueba seria usa decenas de miles y varios regímenes de mercado.

## Noticias

El archivo opcional usa las columnas: timestamp, currency, impact, event.
Si existe un evento HIGH para una de las monedas del par a ±15 minutos de la vela actual, la decisión se convierte en **NO OPERAR**. Esta primera versión usa un CSV para que la fuente del calendario sea verificable y no dependa de scraping frágil.

## Instalación

```bash
git clone https://github.com/otonielcruzmiguel-ux/binary-options-5m-ai.git
cd binary-options-5m-ai
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Entrenar

```bash
python -m src.binary5m train TU_ARCHIVO.csv --payout 0.80
```

El programa imprime accuracy total, número de señales que superan el umbral, accuracy de esas señales y resultado teórico en unidades.

## Generar una señal

```bash
python -m src.binary5m signal TU_ARCHIVO.csv --pair EURUSD --news data/news_events.csv
```

Puede devolver **SUBE**, **BAJA** o **NO OPERAR**.

## Interfaz

```bash
streamlit run app.py
```

## Cómo evaluar correctamente

No juzgues el modelo por una sola sesión. La prueba importante es rendimiento fuera de muestra. Para payout de 80%, el break-even idealizado es 55.56% de aciertos: 1/(1+0.80). El programa añade margen al umbral de entrada, pero eso no implica que la probabilidad del modelo esté perfectamente calibrada.

Antes de cualquier uso real conviene añadir walk-forward validation, calibración probabilística, costos/latencia, datos de varios meses/años, métricas por horario y activo, y una fuente en vivo fiable tanto para mercado como para calendario económico.