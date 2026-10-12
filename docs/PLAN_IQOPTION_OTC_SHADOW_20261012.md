# Plan de corrección: aprendizaje 24/7 y mercados IQ Option (solo simulación)

Estado: propuesta técnica; **no desplegar ni activar operaciones**.

## Diagnóstico confirmado
- `app.py` define `MARKETS` con símbolos de Deriv y consulta `history_store.load(symbol)`.
- `src/history_store.py` persiste velas con clave `(symbol,timestamp)`, sin proveedor ni tipo de mercado.
- `app.py` lee `signals`, `signals_1m` y `rejected_signals` desde SQLite; las métricas de señales aceptadas no incluyen necesariamente las simuladas.
- Logs recientes: predominan avisos `CUARENTENA`; el despliegue actual no contiene integración IQ Option confirmada.

## Implementación propuesta
1. **Identidad del instrumento:** clave `(provider, market_type, symbol, timestamp)`, donde provider=iqoption y market_type=regular|otc. Migración aditiva, sin borrar ni mezclar histórico Deriv. Conservar tablas antiguas para compatibilidad hasta verificar migración.
2. **Fuente IQ Option:** adaptador de datos **solo lectura**, con `fetch_candles(instrument, granularity=60)`, `healthcheck` y registro de frescura. No habilitar autenticación ni trading hasta verificar fuente autorizada, condiciones de uso y disponibilidad de velas OTC. Si no hay fuente fiable: `DATA_UNAVAILABLE`, nunca sustituir por Deriv.
3. **Shadow learning:** generar y guardar predicción hipotética por cada cierre de vela, incluso cuando el filtro rechace la señal; registrar motivo de rechazo, precio/hora de entrada, vencimiento, precio de salida y resultado. No convertir observaciones hipotéticas en señales autorizadas.
4. **Cuarentena:** al recuperar un par, fijar `recovered_at` y exigir un número mínimo configurable de observaciones **posteriores** a esa fecha antes de volver a decidir cuarentena. No reutilizar exclusivamente el historial que provocó el bloqueo. Mantener límites de riesgo existentes.
5. **Separación de métricas:** por proveedor, mercado regular/OTC, símbolo, horizonte 1M/5M y tipo accepted/shadow. Para accuracy: `wins/(wins+losses)`; empates por separado; con denominador cero mostrar N/D.
6. **RESUMEN_8H:** incluir accepted y shadow por separado; contadores de señales generadas, evaluadas, pendientes, descartadas, errores de datos y cuarentenas. Emitir alertas si no se evalúa ninguna shadow en dos bloques consecutivos con datos frescos.
7. **SQLite:** usar context managers y cierre explícito de conexiones de lectura; evitar escrituras desde el dashboard.
8. **Control de seguridad:** feature flags `IQOPTION_DATA_ENABLED=false`, `OTC_ENABLED=false`, `TRADING_ENABLED=false` por defecto; ningún endpoint de órdenes, compras o ventas.

## Pruebas obligatorias antes de cualquier despliegue
- Mercado regular y OTC con el mismo par nunca comparten velas ni resultados.
- Datos OTC ausentes no se sustituyen por cotizaciones regulares.
- Una señal en cuarentena se evalúa en shadow al vencer.
- Un par recuperado no se bloquea por el mismo conjunto de observaciones anteriores a `recovered_at`.
- Un resultado se contabiliza exactamente una vez (idempotencia).
- Sin señales decididas: accuracy N/D, no 0% ni 100%.
- Reinicio del worker no pierde observaciones pendientes ni duplica evaluaciones.
- No existe ruta ejecutable de trading ni cambio automático de modelo por esta propuesta.

## Criterio de aceptación
Dos resúmenes de ocho horas consecutivos con actividad shadow verificable, sin errores persistentes y métricas separadas; después comparar rendimiento fuera de muestra. No prometer aumento de accuracy.

## Alcance actual
Este documento es una especificación revisable. No contiene implementación ni valida la conectividad IQ Option. No fusionar a `main` ni desplegar sin revisión explícita.
