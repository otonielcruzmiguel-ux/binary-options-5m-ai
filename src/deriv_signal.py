from __future__ import annotations
import argparse, asyncio, tempfile
from src.deriv_market import candles
from src.binary5m import signal

async def run(symbol, pair, model, news, count):
    df=await candles(symbol,count,60)
    with tempfile.NamedTemporaryFile(suffix=".csv",delete=False,mode="w") as f:
        df.to_csv(f.name,index=False)
        path=f.name
    signal(path,model,news,pair)

def main():
    ap=argparse.ArgumentParser(description="Señal 5m usando datos públicos oficiales de Deriv")
    ap.add_argument("--symbol",default="frxEURUSD")
    ap.add_argument("--pair",default="EURUSD")
    ap.add_argument("--model",default="models/model.joblib")
    ap.add_argument("--news")
    ap.add_argument("--count",type=int,default=500)
    a=ap.parse_args()
    asyncio.run(run(a.symbol,a.pair,a.model,a.news,a.count))
if __name__=="__main__": main()
