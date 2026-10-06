from __future__ import annotations
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

FEATURES = [
    "ret1","ret3","ret5","ema_gap","rsi14","atr14","bb_pos",
    "range_pct","body_pct","volatility10","hour_sin","hour_cos",
    "support_dist","resistance_dist","breakout_up","breakout_down",
    "retest_up","retest_down","support_bounce","resistance_bounce",
    "ema_trend","ema_cross","rsi_support","rsi_resistance",
    "rsi_bull_div","rsi_bear_div","bb_rsi_buy","bb_rsi_sell",
    "trend_pullback","confluence"
]

def load_candles(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"timestamp","open","high","low","close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    for c in ["open","high","low","close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["open","high","low","close"])

def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def make_features(df: pd.DataFrame, horizon: int = 5, with_target: bool = True) -> pd.DataFrame:
    x = df.copy()
    c = x["close"]
    x["ret1"] = c.pct_change(1)
    x["ret3"] = c.pct_change(3)
    x["ret5"] = c.pct_change(5)
    ema9 = c.ewm(span=9, adjust=False).mean()
    ema21 = c.ewm(span=21, adjust=False).mean()
    x["ema_gap"] = (ema9 - ema21) / c
    x["rsi14"] = rsi(c, 14) / 100
    prev = c.shift(1)
    tr = pd.concat([(x.high-x.low).abs(), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x["atr14"] = tr.rolling(14).mean() / c
    ma20 = c.rolling(20).mean()
    sd20 = c.rolling(20).std()
    x["bb_pos"] = (c - ma20) / (2 * sd20.replace(0, np.nan))
    x["range_pct"] = (x.high-x.low) / c
    x["body_pct"] = (x.close-x.open) / x.open
    x["volatility10"] = x["ret1"].rolling(10).std()

    # Estrategias técnicas convertidas en variables numéricas para ML.
    support = x["low"].shift(1).rolling(20).min()
    resistance = x["high"].shift(1).rolling(20).max()
    atr_abs = tr.rolling(14).mean().replace(0, np.nan)
    x["support_dist"] = (c-support)/atr_abs
    x["resistance_dist"] = (resistance-c)/atr_abs
    x["breakout_up"] = (c > resistance).astype(float)
    x["breakout_down"] = (c < support).astype(float)
    tol = atr_abs * .25
    x["retest_up"] = ((c > resistance.shift(1)) & (x["low"] <= resistance.shift(1)+tol)).astype(float)
    x["retest_down"] = ((c < support.shift(1)) & (x["high"] >= support.shift(1)-tol)).astype(float)
    x["support_bounce"] = ((x["low"] <= support+tol) & (c > x["open"])).astype(float)
    x["resistance_bounce"] = ((x["high"] >= resistance-tol) & (c < x["open"])).astype(float)
    x["ema_trend"] = np.sign(ema9-ema21)
    prev_gap=(ema9-ema21).shift(1)
    x["ema_cross"] = np.where((ema9>ema21)&(prev_gap<=0),1,np.where((ema9<ema21)&(prev_gap>=0),-1,0))
    rsi_raw=rsi(c,14)
    x["rsi_support"] = ((rsi_raw<35) & (x["support_dist"]<.5)).astype(float)
    x["rsi_resistance"] = ((rsi_raw>65) & (x["resistance_dist"]<.5)).astype(float)
    x["rsi_bull_div"] = ((c<c.shift(5)) & (rsi_raw>rsi_raw.shift(5))).astype(float)
    x["rsi_bear_div"] = ((c>c.shift(5)) & (rsi_raw<rsi_raw.shift(5))).astype(float)
    lower=ma20-2*sd20
    upper=ma20+2*sd20
    x["bb_rsi_buy"] = ((c<=lower) & (rsi_raw<35)).astype(float)
    x["bb_rsi_sell"] = ((c>=upper) & (rsi_raw>65)).astype(float)
    near_ema=(c-ema21).abs()/atr_abs
    x["trend_pullback"] = np.where((ema9>ema21)&(near_ema<.5),1,np.where((ema9<ema21)&(near_ema<.5),-1,0))
    bullish=x["breakout_up"]+x["retest_up"]+x["support_bounce"]+(x["ema_trend"]>0).astype(float)+x["rsi_support"]+x["rsi_bull_div"]+x["bb_rsi_buy"]+(x["trend_pullback"]>0).astype(float)
    bearish=x["breakout_down"]+x["retest_down"]+x["resistance_bounce"]+(x["ema_trend"]<0).astype(float)+x["rsi_resistance"]+x["rsi_bear_div"]+x["bb_rsi_sell"]+(x["trend_pullback"]<0).astype(float)
    x["confluence"]=(bullish-bearish)/8.0

    hour = x.timestamp.dt.hour + x.timestamp.dt.minute / 60
    x["hour_sin"] = np.sin(2*np.pi*hour/24)
    x["hour_cos"] = np.cos(2*np.pi*hour/24)
    if with_target:
        future = c.shift(-horizon)
        x["target"] = (future > c).astype(float)
        x.loc[future.isna(), "target"] = np.nan
    return x

def news_block(timestamp, news_path=None, currencies=("EUR","USD"), minutes=15):
    if not news_path or not Path(news_path).exists():
        return False, "Sin calendario de noticias cargado"
    n = pd.read_csv(news_path)
    if n.empty:
        return False, "Sin eventos"
    n["timestamp"] = pd.to_datetime(n["timestamp"], utc=True)
    n["impact"] = n["impact"].astype(str).str.upper()
    n["currency"] = n["currency"].astype(str).str.upper()
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    mask = n["currency"].isin(currencies) & n["impact"].isin(["HIGH","ALTO"])
    delta = (n["timestamp"] - ts).abs().dt.total_seconds()/60
    hit = n[mask & (delta <= minutes)].sort_values("timestamp")
    if hit.empty:
        return False, f"Sin noticias de alto impacto a ±{minutes} min"
    e = hit.iloc[0]
    return True, f"{e['currency']} {e['event']} ({e['timestamp']})"

def train(csv_path, model_path="models/model.joblib", payout=0.80):
    raw = load_candles(csv_path)
    df = make_features(raw, 5, True).dropna(subset=FEATURES+["target"]).copy()
    if len(df) < 500:
        raise ValueError("Se recomiendan al menos 500 velas; idealmente decenas de miles.")
    cut = int(len(df)*0.70)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    model = HistGradientBoostingClassifier(max_iter=250, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=42)
    model.fit(train_df[FEATURES], train_df["target"].astype(int))
    p = model.predict_proba(test_df[FEATURES])[:,1]
    threshold = max(1/(1+payout)+0.03, 0.58)
    take = (p >= threshold) | (p <= 1-threshold)
    pred = (p >= .5).astype(int)
    y = test_df["target"].astype(int).to_numpy()
    acc_all = accuracy_score(y, pred)
    if take.any():
        wins = (pred[take] == y[take])
        acc_trades = wins.mean()
        pnl_units = np.where(wins, payout, -1.0).sum()
    else:
        acc_trades, pnl_units = np.nan, 0.0
    payload={"model":model,"features":FEATURES,"threshold":threshold,"payout":payout}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, model_path)
    print(f"Velas útiles: {len(df)} | Test: {len(test_df)}")
    print(f"Accuracy total: {acc_all:.3f}")
    print(f"Umbral para operar: {threshold:.3f} | Señales test: {int(take.sum())}")
    print(f"Accuracy señales: {acc_trades:.3f}" if take.any() else "No hubo señales con ese umbral")
    print(f"Resultado teórico: {pnl_units:.2f} unidades (payout={payout:.0%})")
    print(classification_report(y, pred, digits=3))

def signal(csv_path, model_path="models/model.joblib", news_path=None, pair="EURUSD"):
    payload=joblib.load(model_path)
    raw=load_candles(csv_path)
    df=make_features(raw, 5, False).dropna(subset=payload["features"])
    if df.empty: raise ValueError("No hay suficientes velas para calcular indicadores.")
    row=df.iloc[-1]
    p=float(payload["model"].predict_proba(df[payload["features"]].iloc[[-1]])[0,1])
    th=float(payload["threshold"])
    currencies=(pair[:3].upper(), pair[3:6].upper()) if len(pair)>=6 else ("EUR","USD")
    blocked, why=news_block(row.timestamp, news_path, currencies)
    if blocked: decision="NO OPERAR"
    elif p>=th: decision="SUBE"
    elif p<=1-th: decision="BAJA"
    else: decision="NO OPERAR"
    confidence=max(p,1-p)
    print(f"{pair} | {row.timestamp} | {decision} | confianza modelo {confidence:.1%}")
    print(f"P(SUBE)={p:.1%} | P(BAJA)={1-p:.1%} | Noticias: {why}")

def main():
    ap=argparse.ArgumentParser(description="Investigación de señales binarias a 5 minutos (demo)")
    sub=ap.add_subparsers(dest="cmd", required=True)
    t=sub.add_parser("train"); t.add_argument("csv"); t.add_argument("--model",default="models/model.joblib"); t.add_argument("--payout",type=float,default=.80)
    s=sub.add_parser("signal"); s.add_argument("csv"); s.add_argument("--model",default="models/model.joblib"); s.add_argument("--news"); s.add_argument("--pair",default="EURUSD")
    a=ap.parse_args()
    if a.cmd=="train": train(a.csv,a.model,a.payout)
    else: signal(a.csv,a.model,a.news,a.pair)
if __name__=="__main__": main()
