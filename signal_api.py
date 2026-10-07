import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import joblib, pandas as pd
from src.history_store import load
from src.binary5m import FEATURES, make_features

MARKETS={"EUR/USD":"frxEURUSD","GBP/USD":"frxGBPUSD","USD/JPY":"frxUSDJPY","AUD/USD":"frxAUDUSD","USD/CAD":"frxUSDCAD","USD/CHF":"frxUSDCHF","EUR/JPY":"frxEURJPY","GBP/JPY":"frxGBPJPY","EUR/GBP":"frxEURGBP","AUD/JPY":"frxAUDJPY"}

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        is_candles=self.path.startswith("/candles/")
        if not (self.path.startswith("/signal/") or is_candles):
            self.send_response(404); self.end_headers(); return
        market=self.path.split("/candles/" if is_candles else "/signal/",1)[1].replace("%2F","/")
        symbol=MARKETS.get(market)
        if not symbol:
            self.send_response(404); self.end_headers(); return
        try:
            if is_candles:
                raw=load(symbol,300)
                rows=[{"timestamp":pd.Timestamp(r.timestamp).isoformat(),"open":r.open,"high":r.high,"low":r.low,"close":r.close,"volume":r.volume} for r in raw.itertuples()]
                body=json.dumps({"market":market,"candles":rows}).encode()
                self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(body); return
            path=Path(f"/data/model_{symbol}.joblib")
            if not path.exists(): raise RuntimeError("modelo aún no disponible")
            payload=joblib.load(path)
            raw=load(symbol,300)
            if raw.empty: raise RuntimeError("histórico local aún no disponible")
            feat=make_features(raw,5,False).dropna(subset=FEATURES)
            p=float(payload["model"].predict_proba(feat[payload["features"]].iloc[[-1]])[0,1])
            th=float(payload.get("threshold",.59))
            decision="SUBE" if p>=th else ("BAJA" if p<=1-th else "NO OPERAR")
            body=json.dumps({"market":market,"symbol":symbol,"decision":decision,"confidence":max(p,1-p),"price":float(raw.iloc[-1].close),"calculated_at":pd.Timestamp.now(tz="UTC").isoformat(),"source":"worker-24x7"}).encode()
            self.send_response(200); self.send_header("Content-Type","application/json"); self.send_header("Cache-Control","no-store"); self.end_headers(); self.wfile.write(body)
        except Exception as e:
            body=json.dumps({"error":str(e)}).encode(); self.send_response(503); self.send_header("Content-Type","application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self,*args): pass

if __name__=="__main__":
    ThreadingHTTPServer(("0.0.0.0",8090),Handler).serve_forever()
