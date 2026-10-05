from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
import pandas as pd
import websockets

PUBLIC_WS = "wss://api.derivws.com/trading/v1/options/ws/public"

async def request(payload: dict) -> dict:
    async with websockets.connect(PUBLIC_WS, ping_interval=20, ping_timeout=20) as ws:
        await ws.send(json.dumps(payload))
        while True:
            data=json.loads(await ws.recv())
            if data.get("error"):
                raise RuntimeError(data["error"].get("message", str(data["error"])))
            if data.get("msg_type") in ("candles","active_symbols","history"):
                return data

async def active_symbols():
    return await request({"active_symbols":"brief","product_type":"basic","req_id":1})

async def candles(symbol: str, count: int=5000, granularity: int=60) -> pd.DataFrame:
    data=await request({
        "ticks_history":symbol,
        "end":"latest",
        "count":count,
        "style":"candles",
        "granularity":granularity,
        "adjust_start_time":1,
        "req_id":2,
    })
    rows=data.get("candles",[])
    if not rows:
        raise RuntimeError("Deriv no devolvió velas para ese símbolo.")
    df=pd.DataFrame(rows)
    df=df.rename(columns={"epoch":"timestamp"})
    df["timestamp"]=pd.to_datetime(df["timestamp"],unit="s",utc=True)
    for c in ["open","high","low","close"]:
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df["volume"]=0
    return df[["timestamp","open","high","low","close","volume"]].dropna()

async def save(symbol: str, output: str, count: int):
    df=await candles(symbol,count,60)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(output,index=False)
    print(f"Guardadas {len(df)} velas de {symbol} en {output}")
    print(df.tail())

def main():
    ap=argparse.ArgumentParser(description="Descarga velas oficiales de Deriv (API pública)")
    ap.add_argument("--symbol",default="frxEURUSD")
    ap.add_argument("--count",type=int,default=5000)
    ap.add_argument("--output",default="data/deriv_candles.csv")
    a=ap.parse_args()
    asyncio.run(save(a.symbol,a.output,a.count))

if __name__=="__main__":
    main()
