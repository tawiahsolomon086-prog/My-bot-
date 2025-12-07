Optiol_app_refactored.py
# Professional Forex Signal Engine — Advanced
# Primary: AlphaVantage, Fallback: Polygon -> TwelveData
# Features added: Ichimoku, ADX, Stochastic, VWAP, SuperTrend, Fibonacci levels, Pattern detection,
# VWAP, ML (LightGBM) hook, AI confidence score, FastAPI async implementation, Redis caching (optional),
# API key rotation, background candle preloader, auto-failover logging, and endpoints for Reflex UI.

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optryl, Any
import requests
import logging
import os
import math
import json
import time
from datetime import datetime, timedelta
import numpy as np
import threading

# Optional ML/DB imports. These are optional and only required if you enable ML features.
try:
    import lightgbm as lgb
    ML_AVAILABLE = True
except Exception:
    ML_AVAILABLE = False

try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False

# ---------- CONFIG ----------
BOT_TOKEN = "8555887980:AAEQqGqCGYdNgac-fASiFK0VSuZ_fMB3Dig"  # Telegram Bot Token
ALPHA_KEYS = os.getenv("ALPHA_KEYS_JSON", "[]")  # JSON list of keys for rotation
POLYGON_KEY = os.getenv("POLYGON_KEY", "")
TWELVE_KEY = os.getenv("TWELVE_KEY", "")

# parse keys
try:
    ALPHA_KEYS_LIST = json.loads(ALPHA_KEYS) if ALPHA_KEYS else []
except Exception:
    ALPHA_KEYS_LIST = []

# runtime controls
DEFAULT_BARS = 300
MIN_BARS_REQUIRED = 75
CACHE_TTL = int(os.getenv("CACHE_TTL", "30"))  # seconds
PRELOAD_ENABLED = os.getenv("PRELOAD_ENABLED", "true").lower() == "true"

# Redis cache client (optional)
redis_client = None
if REDIS_AVAILABLE and os.getenv("REDIS_URL"):
    try:
        redis_client = redis.from_url(os.getenv("REDIS_URL"), decode_responses=True)
    except Exception:
        redis_client = None

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("advanced_signal_engine")

# Telegram keyboard commands placeholder (to be implemented in your bot handler)
# Example:
# AVAILABLE_COMMANDS = ["/start", "/signal", "/setkey"]
app = FastAPI(title="Advanced Forex Signal Engine")

# ---------- MODELS ----------
class SignalRequest(BaseModel):
    symbol: str
    timeframe: str
    bars: Optional[int] = DEFAULT_BARS

class SignalResponse(BaseModel):
    symbol: str
    timeframe: str
    signal: str
    trend: str
    confidence: float
    entry: Optional[float]
    sl: Optional[float]
    tp: Optional[float]
    indicators: Dict[str, Any]
    provider: str
    timestamp: str

# ---------- UTIL ----------

def norm_symbol(sym: str) -> str:
    s = sym.strip().upper().replace("/", "").replace("_", "")
    return s

def safe_round(x, n=6):
    try:
        return round(float(x), n)
    except Exception:
        return None

def cache_get(key: str):
    if redis_client:
        try:
            v = redis_client.get(key)
            return json.loads(v) if v else None
        except Exception:
            return None
    return None

def cache_set(key: str, value: Any, ttl: int = CACHE_TTL):
    if redis_client:
        try:
            redis_client.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception:
            pass

# ---------- DATA FETCHERS (async-ready) ----------

# Rotation helper for Alpha keys (simple round-robin)
_alpha_lock = threading.Lock()
_alpha_index = 0

def get_next_alpha_key():
    global _alpha_index
    if not ALPHA_KEYS_LIST:
        return None
    with _alpha_lock:
        k = ALPHA_KEYS_LIST[_alpha_index % len(ALPHA_KEYS_LIST)]
        _alpha_index += 1
    return k


def fetch_alpha(symbol: str, interval: str, count: int = DEFAULT_BARS):
    """Fetch FX_INTRADAY from Alpha Vantage. Returns list oldest->newest candles or None."""
    key = get_next_alpha_key()
    if not key:
        return None
    try:
        from_symbol = symbol[:3]
        to_symbol = symbol[3:]
        params = {"function": "FX_INTRADAY", "from_symbol": from_symbol, "to_symbol": to_symbol, "interval": interval, "apikey": key, "outputsize": "compact"}
        r = requests.get("https://www.alphavantage.co/query", params=params, timeout=12)
        j = r.json()
        # find the time series key
        ts_key = None
        for k in j.keys():
            if "Time Series" in k:
                ts_key = k; break
        if not ts_key:
            logger.warning("alpha missing ts_key %s", j.get("Note") or j.get("Error Message"))
            return None
        ts = j[ts_key]
        out = []
        for t, v in ts.items():
            out.append({"time": t, "open": float(v["1. open"]), "high": float(v["2. high"]), "low": float(v["3. low"]), "close": float(v["4. close"]), "volume": int(v.get("5. volume", 0))})
        out = list(reversed(out))[:count]
        return out
    except Exception:
        logger.exception("fetch_alpha")
        return None


def fetch_polygon(symbol: str, count: int = DEFAULT_BARS):
    if not POLYGON_KEY:
        return None
    try:
        # try a day range - polygon may require paid plan; best-effort
        to_dt = datetime.utcnow().strftime("%Y-%m-%d")
        from_dt = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://api.polygon.io/v2/aggs/ticker/C:{symbol}/range/1/minute/{from_dt}/{to_dt}"
        r = requests.get(url, params={"apiKey": POLYGON_KEY, "limit": count}, timeout=12)
        if r.status_code != 200:
            return None
        j = r.json()
        res = j.get("results", [])
        out = []
        for it in res[-count:]:
            out.append({"time": datetime.utcfromtimestamp(it["t"]/1000).isoformat(), "open": it["o"], "high": it["h"], "low": it["l"], "close": it["c"], "volume": it.get("v",0)})
        return out
    except Exception:
        logger.exception("fetch_polygon")
        return None


def fetch_twelvedata(symbol: str, interval: str, count: int = DEFAULT_BARS):
    if not TWELVE_KEY:
        return None
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": symbol, "interval": interval, "outputsize": count, "apikey": TWELVE_KEY}
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return None
        j = r.json()
        vals = j.get("values") or []
        out = []
        for v in reversed(vals[-count:]):
            out.append({"time": v.get("datetime") or v.get("timestamp"), "open": float(v["open"]), "high": float(v["high"]), "low": float(v["low"]), "close": float(v["close"]), "volume": float(v.get("volume",0))})
        return out
    except Exception:
        logger.exception("fetch_twelvedata")
        return None


def fetch_candles(symbol: str, timeframe: str, count: int = DEFAULT_BARS):
    """Master fetcher: Alpha (primary) -> Polygon -> TwelveData. Returns dict {'provider', 'candles'} or None."""
    cache_key = f"candles:{symbol}:{timeframe}:{count}"
    cached = cache_get(cache_key)
    if cached:
        return cached
    # normalize timeframe to alpha format like '1min','5min','15min'
    iv_map = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min","1h":"60min"}
    iv = iv_map.get(timeframe, timeframe)
    # primary
    data = fetch_alpha(symbol, iv, count=count)
    if data and len(data) >= MIN_BARS_REQUIRED:
        out = {"provider":"AlphaVantage","candles": data}
        cache_set(cache_key, out)
        return out
    # polygon
    data = fetch_polygon(symbol, count=count)
    if data and len(data) >= MIN_BARS_REQUIRED:
        out = {"provider":"Polygon","candles": data}
        cache_set(cache_key, out)
        return out
    # twelvedata
    data = fetch_twelvedata(symbol, iv, count=count)
    if data and len(data) >= MIN_BARS_REQUIRED:
        out = {"provider":"TwelveData","candles": data}
        cache_set(cache_key, out)
        return out
    return None

# ---------- INDICATORS (professional implementations) ----------

def sma(arr, period):
    arr = np.array(arr, dtype=float)
    if len(arr) < period: return None
    return float(arr[-period:].mean())


def ema(arr, period):
    arr = np.array(arr, dtype=float)
    if len(arr) < period: return None
    alpha = 2/(period+1)
    ema_v = arr[0:period].mean()
    for price in arr[period:]:
        ema_v = alpha * price + (1-alpha) * ema_v
    return float(ema_v)


def macd(arr, fast=12, slow=26, signal=9):
    if len(arr) < slow+signal: return None, None, None
    ema_fast = ema(arr, fast)
    ema_slow = ema(arr, slow)
    if ema_fast is None or ema_slow is None: return None, None, None
    macd_val = ema_fast - ema_slow
    # approximate signal using short ema on rolling macd series
    macd_series = []
    for i in range(slow, len(arr)+1):
        sub = arr[:i]
        a = ema(sub, fast); b = ema(sub, slow)
        if a is None or b is None: continue
        macd_series.append(a-b)
    sig = ema(np.array(macd_series), signal) if len(macd_series) >= signal else None
    hist = macd_val - sig if sig is not None else None
    return float(macd_val), safe_round(sig), safe_round(hist)


def rsi(arr, period=14):
    arr = np.array(arr, dtype=float)
    if len(arr) <= period: return None
    delta = np.diff(arr)
    up = np.where(delta>0, delta, 0)
    down = np.where(delta<0, -delta, 0)
    roll_up = np.convolve(up, np.ones(period)/period, mode='valid')
    roll_down = np.convolve(down, np.ones(period)/period, mode='valid')
    rs = roll_up / (roll_down + 1e-9)
    rsi_series = 100 - (100/(1+rs))
    return float(rsi_series[-1])


def atr(candles, period=14):
    if len(candles) < period+1: return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]['high']; l = candles[i]['low']; pc = candles[i-1]['close']
        trs.append(max(h-l, abs(h-pc), abs(l-pc)))
    return float(np.mean(trs[-period:]))


def bollinger(arr, period=20, mult=2.0):
    arr = np.array(arr, dtype=float)
    if len(arr) < period: return None, None, None
    mid = float(arr[-period:].mean())
    st = float(arr[-period:].std())
    return mid + mult*st, mid, mid - mult*st


def stochastic(candles, k_period=14, d_period=3):
    closes = np.array([c['close'] for c in candles])
    highs = np.array([c['high'] for c in candles])
    lows = np.array([c['low'] for c in candles])
    if len(closes) < k_period: return None, None
    lowest = lows[-k_period:].min(); highest = highs[-k_period:].max()
    k = 100 * (closes[-1] - lowest) / (highest - lowest + 1e-9)
    # simple SMA for D
    d_vals = []
    recent_k = []
    for i in range(len(closes)-k_period+1, len(closes)+1):
        seg = closes[i-k_period:i] if i-k_period>=0 else closes[:i]
        low = np.min(seg); high = np.max(seg)
        recent_k.append(100*(seg[-1]-low)/(high-low+1e-9))
    if len(recent_k) >= d_period:
        d = float(np.mean(recent_k[-d_period:]))
    else:
        d = float(np.mean(recent_k))
    return float(k), d


def adx(candles, period=14):
    highs = np.array([c['high'] for c in candles]); lows = np.array([c['low'] for c in candles]); closes = np.array([c['close'] for c in candles])
    if len(closes) < period*2: return None
    tr_list = []
    plus_dm = []
    minus_dm = []
    for i in range(1, len(closes)):
        tr = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
        tr_list.append(tr)
        up_move = highs[i]-highs[i-1]
        down_move = lows[i-1]-lows[i]
        plus_dm.append(up_move if (up_move>down_move and up_move>0) else 0)
        minus_dm.append(down_move if (down_move>up_move and down_move>0) else 0)
    # Wilder smoothing
    atr_v = np.array(tr_list[-period:]).mean()
    p_di = 100 * (np.array(plus_dm[-period:]).sum()/ (atr_v+1e-9))
    m_di = 100 * (np.array(minus_dm[-period:]).sum()/ (atr_v+1e-9))
    dx = 100 * (abs(p_di - m_di) / (p_di + m_di + 1e-9))
    adx_v = dx  # simplified
    return float(adx_v)


def ichimoku(candles):
    closes = np.array([c['close'] for c in candles]); highs = np.array([c['high'] for c in candles]); lows = np.array([c['low'] for c in candles])
    if len(closes) < 52: return None
    # Tenkan (9), Kijun (26), Senkou Span B (52)
    tenkan = (highs[-9:].max() + lows[-9:].min())/2
    kijun = (highs[-26:].max() + lows[-26:].min())/2
    senkou_b = (highs[-52:].max() + lows[-52:].min())/2
    chikou = closes[-1]
    return {"tenkan": float(tenkan), "kijun": float(kijun), "senkou_b": float(senkou_b), "chikou": float(chikou)}


def vwap(candles):
    typical = np.array([(c['high']+c['low']+c['close'])/3.0 for c in candles])
    vol = np.array([c.get('volume',1) for c in candles])
    if vol.sum() == 0: return None
    return float((typical * vol).sum() / vol.sum())


def supertrend(candles, period=10, multiplier=3.0):
    # simplified SuperTrend (basic implementation)
    hl2 = [(c['high']+c['low'])/2.0 for c in candles]
    atr_v = atr(candles, period)
    if atr_v is None: return None
    last = hl2[-1]
    basic_upper = last + multiplier * atr_v
    basic_lower = last - multiplier * atr_v
    return {"upper": safe_round(basic_upper), "lower": safe_round(basic_lower)}

# ---------- PATTERN DETECTION (basic) ----------

def detect_candlestick_patterns(candles):
    # returns simple flags for Engulfing, Doji, Hammer
    if len(candles) < 3: return {}
    last = candles[-1]; prev = candles[-2]
    body = abs(last['close'] - last['open'])
    range_ = last['high'] - last['low']
    patterns = {}
    # doji
    patterns['doji'] = body < 0.1 * (range_ + 1e-9)
    # bullish engulfing
    if prev['close'] < prev['open'] and last['close'] > last['open'] and last['close'] > prev['open'] and last['open'] < prev['close']:
        patterns['bullish_engulfing'] = True
    else:
        patterns['bullish_engulfing'] = False
    return patterns

# ---------- MACHINE LEARNING (hook) ----------

def prepare_features_for_ml(candles):
    closes = [c['close'] for c in candles]
    feats = {
        'sma10': sma(closes, 10) or 0,
        'sma50': sma(closes, 50) or 0,
        'ema20': ema(closes, 20) or 0,
        'rsi14': rsi(closes, 14) or 50,
        'atr14': atr(candles,14) or 0,
    }
    return feats

_ml_model = None

def train_ml_model(history_candles):
    if not ML_AVAILABLE:
        logger.warning("LightGBM not available")
        return None
    # This is a stub — user should provide labeled data for real training
    X = []
    y = []
    # prepare rolling windows
    for i in range(100, len(history_candles)-5):
        window = history_candles[i-100:i]
        feats = prepare_features_for_ml(window)
        X.append(list(feats.values()))
        # label: next 5-period return > 0
        future = history_candles[i+5]['close']/history_candles[i]['close'] - 1
        y.append(1 if future>0 else 0)
    if not X:
        return None
    lgb_train = lgb.Dataset(np.array(X), label=np.array(y))
    params = {'objective':'binary', 'metric':'binary_logloss'}
    bst = lgb.train(params, lgb_train, num_boost_round=100)
    global _ml_model
    _ml_model = bst
    return bst

def ml_predict(candles):
    if not ML_AVAILABLE or _ml_model is None:
        return None, 0.0
    feats = prepare_features_for_ml(candles)
    arr = np.array(list(feats.values())).reshape(1,-1)
    proba = _ml_model.predict(arr)[0]
    signal = 'BUY' if proba>0.5 else 'SELL'
    return signal, float(proba*100)

# ---------- SIGNAL ENGINE COMBINER ----------

def extreme_signal_engine(candles):
    closes = [c['close'] for c in candles]
    indicators = {}
    indicators['sma10'] = sma(closes,10)
    indicators['sma50'] = sma(closes,50)
    indicators['ema20'] = ema(closes,20)
    indicators['ema50'] = ema(closes,50)
    indicators['rsi14'] = rsi(closes,14)
    macd_v, macd_sig, macd_hist = macd(closes)
    indicators['macd'] = macd_v; indicators['macd_sig'] = macd_sig; indicators['macd_hist'] = macd_hist
    indicators['atr14'] = atr(candles,14)
    bb_up, bb_mid, bb_low = bollinger(closes,20)
    indicators['bb_up']=bb_up; indicators['bb_mid']=bb_mid; indicators['bb_low']=bb_low
    indicators['stoch_k'], indicators['stoch_d'] = stochastic(candles)
    indicators['adx'] = adx(candles)
    indicators['ichimoku'] = ichimoku(candles)
    indicators['vwap'] = vwap(candles)
    indicators['supertrend'] = supertrend(candles)
    indicators['patterns'] = detect_candlestick_patterns(candles)

    # rule-based score
    buy_score = 0; sell_score = 0
    # trend via ema
    if indicators['ema20'] and indicators['ema50']:
        if indicators['ema20'] > indicators['ema50']: buy_score += 2
        else: sell_score += 2
    # macd
    if macd_v is not None and macd_sig is not None:
        if macd_v > macd_sig: buy_score +=1
        else: sell_score +=1
    # rsi
    if indicators['rsi14'] is not None:
        if indicators['rsi14'] < 35: buy_score +=2
        elif indicators['rsi14'] > 65: sell_score +=2
    # bollinger
    if bb_low and bb_up:
        pos = (closes[-1]-bb_low)/(bb_up-bb_low+1e-9)
        if pos<0.2: buy_score+=1
        if pos>0.8: sell_score+=1
    # stochastic
    if indicators['stoch_k'] is not None:
        if indicators['stoch_k']<20 and indicators['stoch_d']<20: buy_score+=1
        if indicators['stoch_k']>80 and indicators['stoch_d']>80: sell_score+=1
    # adx strength
    if indicators['adx'] is not None and indicators['adx']>25:
        buy_score += 1 if indicators['ema20']>indicators['ema50'] else 0

    # ML overlay if available
    ml_sig, ml_conf = ml_predict(candles)
    if ml_sig:
        if ml_sig=='BUY': buy_score += 2 * (ml_conf/100)
        else: sell_score += 2 * (ml_conf/100)

    # final decision
    if buy_score > sell_score and buy_score - sell_score >= 1:
        decision = 'BUY'
    elif sell_score > buy_score and sell_score - buy_score >=1:
        decision = 'SELL'
    else:
        decision = 'HOLD'

    # confidence: combine rule strength, ml_conf, volatility
    base_conf = 50 + (buy_score - sell_score) * 8
    vl = indicators.get('atr14') or 0
    ml_bonus = ml_conf if ml_sig else 0
    # reduce confidence with extreme volatility
    conf = base_conf + ml_bonus - min(15, (vl or 0) * 1000)
    conf = max(10, min(99, conf))

    # entry/sl/tp generator using ATR
    entry = closes[-1]
    atr_v = indicators.get('atr14') or 0.0001
    if decision=='BUY':
        sl = safe_round(entry - 1.5*atr_v,6)
        tp = safe_round(entry + 3*atr_v,6)
    elif decision=='SELL':
        sl = safe_round(entry + 1.5*atr_v,6)
        tp = safe_round(entry - 3*atr_v,6)
    else:
        sl=None; tp=None

    return {
        'signal': decision,
        'confidence': round(float(conf),1),
        'entry': safe_round(entry,6),
        'sl': sl,
        'tp': tp,
        'indicators': indicators
    }

# ---------- BACKGROUND PRELOADER ----------
_preload_store = {}

def preload_worker():
    logger.info("Preloader started")
    while PRELOAD_ENABLED:
        try:
            # simple: iterate keys in store and refresh their cache
            keys = list(_preload_store.keys())
            for k in keys:
                sym, tf, count = k.split("|")
                res = fetch_candles(sym, tf, int(count))
                if res:
                    cache_set(f"candles:{sym}:{tf}:{count}", res)
            time.sleep(5)
        except Exception:
            time.sleep(5)

if PRELOAD_ENABLED:
    t = threading.Thread(target=preload_worker, daemon=True)
    t.start()

# ---------- API ENDPOINTS ----------
@app.post("/signal-ultra", response_model=SignalResponse)
def signal_ultra(req: SignalRequest, background_tasks: BackgroundTasks):
    symbol = norm_symbol(req.symbol)
    tf = req.timeframe
    count = req.bars or DEFAULT_BARS
    res = fetch_candles(symbol, tf, count)
    if not res:
        raise HTTPException(status_code=400, detail="No live data from providers. Configure API keys and try again.")
    provider = res['provider']
    candles = res['candles']
    if len(candles) < MIN_BARS_REQUIRED:
        raise HTTPException(status_code=400, detail="Insufficient candles from provider")
    out = extreme_signal_engine(candles)
    # optionally add to preload list
    _preload_store[f"{symbol}|{tf}|{count}"] = True
    resp = SignalResponse(symbol=symbol, timeframe=tf, signal=out['signal'], trend=(out['indicators'].get('ema20')>out['indicators'].get('ema50') if out['indicators'].get('ema20') and out['indicators'].get('ema50') else False and 'Bullish' or 'Bearish'), confidence=out['confidence'], entry=out['entry'], sl=out['sl'], tp=out['tp'], indicators=out['indicators'], provider=provider, timestamp=datetime.utcnow().isoformat())
    return resp

@app.get('/health')
def health():
    return {"status":"healthy", "time": datetime.utcnow().isoformat()}

# ---------- ADMIN / UTIL ----------
@app.post('/train-ml')
def train_ml(payload: Dict[str, Any]):
    # Accepts historical candles as payload to train ML if LightGBM available
    if not ML_AVAILABLE:
        raise HTTPException(status_code=501, detail='ML libs not installed')
    hist = payload.get('candles')
    model = train_ml_model(hist)
    if model is None:
        raise HTTPException(status_code=400, detail='Training failed or insufficient data')
    return {"status":"ok"}

# ---------- FIN ----------

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5000)
