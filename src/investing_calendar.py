import re, json
from pathlib import Path
from datetime import timezone, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup

URL="https://www.investing.com/economic-calendar/"
DATA=Path("/data")
CACHE=DATA/"investing_news.csv"
STATUS=DATA/"investing_news_status.json"

def _status(**kw):
    STATUS.write_text(json.dumps(kw,ensure_ascii=False))

def refresh():
    """Refresh a local snapshot from Investing.com's public economic-calendar page."""
    try:
        r=requests.get(URL,headers={"User-Agent":"Mozilla/5.0 (compatible; BinaryOptionsResearch/1.0)","Accept-Language":"en-US,en;q=0.8"},timeout=12)
        r.raise_for_status()
        soup=BeautifulSoup(r.text,"html.parser")
        text=soup.get_text(" ",strip=True)
        m=re.search(r"(?:Current Time|Hora actual)\s*:?\s*\d{1,2}:\d{2}.*?GMT\s*([+-]\d{1,2})(?::(\d{2}))?",text,re.I)
        hours=int(m.group(1)) if m else 0
        mins=int(m.group(2) or 0) if m else 0
        sign=1 if hours>=0 else -1
        site_tz=timezone(timedelta(hours=hours,minutes=sign*mins))
        rows=[]
        for tr in soup.select("tr.js-event-item, tr[data-event-datetime]"):
            dt=tr.get("data-event-datetime")
            if not dt: continue
            cur=tr.select_one(".flagCur, .left.flagCur")
            sentiment=tr.select_one(".sentiment")
            event=tr.select_one(".event, .event a")
            currency=cur.get_text(" ",strip=True).upper() if cur else ""
            if currency not in {"USD","EUR","GBP","JPY","AUD","CAD","CHF"}: continue
            title=((sentiment.get("title") if sentiment else "") or "").upper()
            icons=len(sentiment.select("i, span")) if sentiment else 0
            high=("HIGH" in title or "ALTA" in title or icons>=3)
            if not high: continue
            ts=pd.Timestamp(dt)
            if ts.tzinfo is None: ts=ts.tz_localize(site_tz)
            ts=ts.tz_convert("UTC")
            rows.append({"timestamp":ts.isoformat(),"currency":currency,"impact":"HIGH","event":event.get_text(" ",strip=True) if event else "Evento económico","source":"Investing.com"})
        out=pd.DataFrame(rows).drop_duplicates(subset=["timestamp","currency","event"]) if rows else pd.DataFrame(columns=["timestamp","currency","impact","event","source"])
        out.to_csv(CACHE,index=False)
        _status(ok=True,updated_at=pd.Timestamp.now(tz="UTC").isoformat(),events=int(len(out)),source=URL)
        return out
    except Exception as e:
        _status(ok=False,updated_at=pd.Timestamp.now(tz="UTC").isoformat(),error=str(e),source=URL)
        return None

def load():
    if not CACHE.exists(): return pd.DataFrame()
    try:
        d=pd.read_csv(CACHE)
        if not d.empty: d["timestamp"]=pd.to_datetime(d["timestamp"],utc=True)
        return d
    except Exception:
        return pd.DataFrame()

def block(timestamp, currencies, before=15, after=30):
    d=load()
    if d.empty: return False,"Calendario Investing no disponible",None
    ts=pd.Timestamp(timestamp)
    if ts.tzinfo is None: ts=ts.tz_localize("UTC")
    hit=d[d["currency"].isin([c.upper() for c in currencies])].copy()
    if hit.empty: return False,"Sin noticias altas para el par",None
    delta=(ts-hit["timestamp"]).dt.total_seconds()/60
    hit=hit[(delta>=-before)&(delta<=after)].copy()
    if hit.empty: return False,"Sin bloqueo por noticias",None
    hit["distance"]=(ts-hit["timestamp"]).abs().dt.total_seconds()
    e=hit.sort_values("distance").iloc[0]
    return True,f"{e['currency']} · {e['event']} · Investing.com",e.to_dict()
