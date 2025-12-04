"""
app.py - Professional Telegram Forex Signal Bot (single-file)
- Reply keyboard
- Full indicators (SMA/EMA, RSI, MACD, Bollinger, ATR-like)
- Support/Resistance & simple candle patterns
- Professional signal with Entry/SL/TP + indicator details
- Bot asks user for OANDA practice/live keys inside Telegram and saves per-user in users.json
- Polling mode -> works on Acode/ArcCode/Termux
- Minimal deps: python-telegram-bot==13.15, requests
"""

import os
import json
import math
import random
import logging
import statistics
from datetime import datetime
from typing import List, Dict, Optional

import requests
from telegram import Bot, Update, ReplyKeyboardMarkup, KeyboardButton, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# -------------------------
# CONFIG
# -------------------------
# Put your Telegram token here or export TELEGRAM_TOKEN env var
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "") or "PASTE_YOUR_TELEGRAM_TOKEN_HERE"

# User data persistence file (stores OANDA keys per chat_id)
USER_FILE = "users.json"

# Defaults
DEFAULT_BARS = 200
MIN_BARS_REQUIRED = 50
USE_REAL_DATA_ONLY = False  # set True to refuse simulated returns if no API keys

# OANDA endpoints (practice/live)
OANDA_PRACTICE_BASE = "https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
OANDA_LIVE_BASE = "https://api-fxtrade.oanda.com/v3/instruments/{instrument}/candles"

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forex_bot")

# -------------------------
# Load / save users
# -------------------------
def load_users() -> Dict:
    if os.path.exists(USER_FILE):
        try:
            return json.load(open(USER_FILE, "r"))
        except Exception:
            return {}
    return {}

def save_users(data: Dict):
    try:
        json.dump(data, open(USER_FILE, "w"), indent=2)
    except Exception:
        logger.exception("Failed saving users")

users = load_users()  # structure: str(chat_id) -> { "oanda_practice": "...", "oanda_live": "..." }

# -------------------------
# In-memory user session state
# -------------------------
session_state: Dict[int, Dict] = {}  # chat_id -> { pair, timeframe, account_choice, awaiting_field }

# -------------------------
# Keyboards (reply keyboard style)
# -------------------------
def main_menu_kb():
    kb = [
        [KeyboardButton("📈 Select Pair"), KeyboardButton("⏱ Select Timeframe")],
        [KeyboardButton("🎯 Request Signal"), KeyboardButton("🔎 Check Trend")],
        [KeyboardButton("🧾 Support/Resistance"), KeyboardButton("📰 Candles/Patterns")],
        [KeyboardButton("⚙️ Settings"), KeyboardButton("ℹ️ Help")],
        [KeyboardButton("🔔 Subscribe"), KeyboardButton("🔕 Unsubscribe")]
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True)

def pairs_kb():
    pairs = ["EURUSD","GBPUSD","USDJPY","AUDUSD","XAUUSD","USDCAD","Back"]
    rows = [pairs[i:i+3] for i in range(0,len(pairs),3)]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def tfs_kb():
    times = ["1m","5m","15m","30m","1h","Back"]
    rows = [times[i:i+3] for i in range(0,len(times),3)]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def account_kb():
    rows = [["Practice","Live","Auto (Prefer Practice)"],["Back"]]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -------------------------
# Utils: instrument formatting
# -------------------------
def normalize_pair(pair: str) -> str:
    # Accept EURUSD, EUR/USD, EUR_USD -> returns "EUR_USD" for OANDA
    s = pair.strip().upper().replace("/","_")
    if "_" not in s and len(s)==6:
        s = s[:3] + "_" + s[3:]
    return s

# -------------------------
# Fetch candles: OANDA -> simulated
# -------------------------
def fetch_oanda_candles(pair: str, timeframe: str, count:int=DEFAULT_BARS, account_type:str="practice", api_key:str=None):
    """
    Returns list of candles oldest->newest as dicts: {time,open,high,low,close,volume}
    timeframe: '1m','5m','15m','30m','1h' maps to OANDA M1,M5,M15,M30,H1
    """
    try:
        if not api_key:
            return None
        gran_map = {"1m":"M1","5m":"M5","15m":"M15","30m":"M30","1h":"H1"}
        gran = gran_map.get(timeframe.lower(), "M5")
        instrument = normalize_pair(pair)
        base = OANDA_PRACTICE_BASE if account_type=="practice" else OANDA_LIVE_BASE
        url = base.format(instrument=instrument)
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {"granularity": gran, "count": count, "price":"M"}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            logger.warning("OANDA HTTP %s %s", r.status_code, r.text[:200])
            return None
        data = r.json()
        if "candles" not in data:
            return None
        out=[]
        for c in data["candles"]:
            out.append({
                "time": c.get("time"),
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "volume": int(c.get("volume",0))
            })
        out = list(reversed(out))  # oldest->newest
        return out
    except Exception:
        logger.exception("fetch_oanda_candles")
        return None

def simulated_candles(base_price=1.1000, bars=DEFAULT_BARS):
    price = base_price
    out=[]
    for _ in range(bars):
        change = random.normalvariate(0, 0.0006)
        o = price
        c = price + change
        h = max(o,c) + abs(random.normalvariate(0, 0.0002))
        l = min(o,c) - abs(random.normalvariate(0, 0.0002))
        out.append({"time":None,"open":round(o,6),"high":round(h,6),"low":round(l,6),"close":round(c,6),"volume":int(1000+random.random()*1000)})
        price = c
    return out

def get_candles_for_user(chat_id:int, pair:str, timeframe:str, account_choice:str="auto"):
    """
    Tries OANDA practice, then OANDA live, else simulated (subject to USE_REAL_DATA_ONLY)
    account_choice: "practice","live","auto"
    """
    user = users.get(str(chat_id),{})
    # if user requested practice/live explicitly use that
    if account_choice=="practice":
        key = user.get("oanda_practice")
        if key:
            data = fetch_oanda_candles(pair,timeframe,api_key=key,account_type="practice")
            if data and len(data)>=MIN_BARS_REQUIRED:
                return data
    elif account_choice=="live":
        key = user.get("oanda_live")
        if key:
            data = fetch_oanda_candles(pair,timeframe,api_key=key,account_type="live")
            if data and len(data)>=MIN_BARS_REQUIRED:
                return data
    else:
        # auto preference practice then live
        key = user.get("oanda_practice")
        if key:
            data = fetch_oanda_candles(pair,timeframe,api_key=key,account_type="practice")
            if data and len(data)>=MIN_BARS_REQUIRED:
                return data
        key = user.get("oanda_live")
        if key:
            data = fetch_oanda_candles(pair,timeframe,api_key=key,account_type="live")
            if data and len(data)>=MIN_BARS_REQUIRED:
                return data

    # fallback: simulated (if allowed)
    if USE_REAL_DATA_ONLY:
        return None
    return simulated_candles(bars=DEFAULT_BARS)

# -------------------------
# Indicators (pure Python)
# -------------------------
def sma(values:List[float], period:int) -> Optional[float]:
    if len(values) < period: return None
    return sum(values[-period:]) / period

def ema(values:List[float], period:int) -> Optional[float]:
    if len(values) < period: return None
    k = 2/(period+1)
    ema_prev = sum(values[:period]) / period  # SMA as starting EMA
    for price in values[period:]:
        ema_prev = (price - ema_prev)*k + ema_prev
    return ema_prev

def rsi(values:List[float], period:int=14) -> Optional[float]:
    if len(values) <= period: return None
    gains = []
    losses = []
    for i in range(1, period+1):
        diff = values[-i] - values[-i-1]
        if diff>0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-diff)
    avg_gain = sum(gains)/period
    avg_loss = sum(losses)/period if sum(losses)!=0 else 1e-9
    rs = avg_gain/avg_loss if avg_loss!=0 else 0
    return 100 - (100/(1+rs))

def stddev(values:List[float], period:int):
    if len(values) < period: return None
    subset = values[-period:]
    return statistics.pstdev(subset)

def bollinger(values:List[float], period:int=20, devs:float=2.0):
    m = sma(values, period)
    sd = stddev(values, period)
    if m is None or sd is None: return None, None, None
    return m + devs*sd, m, m - devs*sd

def macd(values:List[float], fast:int=12, slow:int=26, signal:int=9):
    if len(values) < slow + signal: return None, None, None
    # compute EMA fast and slow arrays - simple approach
    def compute_ema_series(vals, period):
        k = 2/(period+1)
        ema_series = []
        # start at SMA of first period
        ema_prev = sum(vals[:period]) / period
        ema_series.append(ema_prev)
        for price in vals[period:]:
            ema_prev = (price - ema_prev)*k + ema_prev
            ema_series.append(ema_prev)
        return ema_series
    # align lengths: compute full EMAs using tail values
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    macd_val = ema_fast - ema_slow
    # For signal, approximate by computing EMA on MACD series from last (slow..)
    # Fast approach: compute MACD series using rolling EMAs
    macd_series = []
    # Build macd_series by computing ema_fast and ema_slow over rolling window (costly but small)
    for i in range(slow, len(values)+1):
        sub = values[:i]
        ef = ema(sub, fast)
        es = ema(sub, slow)
        if ef is None or es is None: continue
        macd_series.append(ef - es)
    if len(macd_series) < signal:
        return macd_val, None, None
    # compute signal EMA on macd_series
    sig = ema(macd_series, signal)
    hist = macd_val - sig if sig is not None else None
    return macd_val, sig, hist

def atr_like(candles:List[Dict], period:int=14):
    if len(candles) < period+1:
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        return (max(highs)-min(lows))/max(1,len(candles))
    trs=[]
    for i in range(1,len(candles)):
        high = candles[i]['high']; low = candles[i]['low']; prev_close = candles[i-1]['close']
        tr = max(high-low, abs(high-prev_close), abs(low-prev_close))
        trs.append(tr)
    return sum(trs[-period:])/period

# -------------------------
# Support/Resistance simple detection (recent swing highs/lows)
# -------------------------
def detect_support_resistance(candles:List[Dict], lookback:int=100, zone_width=0.0015):
    recent = candles[-lookback:] if len(candles)>lookback else candles
    highs = [(i, c['high']) for i,c in enumerate(recent)]
    lows = [(i, c['low']) for i,c in enumerate(recent)]
    # simple: take top 3 highs and lows
    highs_sorted = sorted(highs, key=lambda x: -x[1])[:3]
    lows_sorted = sorted(lows, key=lambda x: x[1])[:3]
    zones = {"resistance":[], "support":[]}
    for _,h in highs_sorted:
        zones["resistance"].append((round(h*(1-zone_width),6), round(h*(1+zone_width),6)))
    for _,l in lows_sorted:
        zones["support"].append((round(l*(1-zone_width),6), round(l*(1+zone_width),6)))
    return zones

# -------------------------
# Candlestick pattern (very simple): bullish engulfing / bearish engulfing
# -------------------------
def detect_candle_patterns(candles:List[Dict]):
    if len(candles) < 2:
        return []
    last = candles[-1]; prev = candles[-2]
    patterns=[]
    # bullish engulfing
    if prev['close']<prev['open'] and last['close']>last['open'] and last['close']>prev['open'] and last['open']<prev['close']:
        patterns.append("Bullish Engulfing")
    # bearish engulfing
    if prev['close']>prev['open'] and last['close']<last['open'] and last['open']>prev['close'] and last['close']<prev['open']:
        patterns.append("Bearish Engulfing")
    return patterns

# -------------------------
# Signal engine (professional + indicator-based)
# -------------------------
def professional_signal(candles:List[Dict]):
    closes = [c['close'] for c in candles]
    if len(closes) < 20:
        return {"signal":"HOLD","confidence":40,"reason":"insufficient_data"}
    # indicators
    sma10 = sma(closes,10); sma50 = sma(closes,50)
    ema20 = ema(closes,20); ema50 = ema(closes,50)
    rsi_val = rsi(closes,14)
    macd_val, macd_sig, macd_hist = macd(closes)
    bb_up, bb_mid, bb_low = bollinger(closes,20)
    atr = atr_like(candles,14)
    momentum = closes[-1] - closes[-5] if len(closes)>=5 else 0
    # scoring
    buy = 0; sell=0
    # trend
    if sma10 and sma50:
        if sma10 > sma50: buy += 2
        else: sell += 2
    # ema
    if ema20 and ema50:
        if ema20 > ema50: buy += 1
        else: sell += 1
    # momentum
    if momentum>0: buy += 1
    else: sell += 1
    # rsi extremes
    if rsi_val is not None:
        if rsi_val < 30: buy += 2
        elif rsi_val > 70: sell += 2
    # macd
    if macd_val is not None and macd_sig is not None:
        if macd_val > macd_sig: buy += 1
        else: sell += 1
    # bollinger position
    if bb_low and bb_up:
        pos = (closes[-1]-bb_low)/(bb_up-bb_low) if (bb_up-bb_low)!=0 else 0.5
        if pos<0.2: buy+=1
        elif pos>0.8: sell+=1
    # final decision
    if buy > sell:
        base = "BUY"
        raw_conf = 50 + (buy - sell)*10
    elif sell > buy:
        base = "SELL"
        raw_conf = 50 + (sell - buy)*10
    else:
        base = "HOLD"
        raw_conf = 50
    # tweak with volatility and ATR: higher ATR reduces confidence slightly
    conf = max(20, min(99, raw_conf - min(15, atr*1000 if atr else 0)))
    # SL/TP generation using ATR
    entry = round(closes[-1],6)
    if base=="BUY":
        sl = round(entry - atr*1.5,6)
        tp = round(entry + atr*3,6)
    elif base=="SELL":
        sl = round(entry + atr*1.5,6)
        tp = round(entry - atr*3,6)
    else:
        sl=None; tp=None
    indicators = {
        "sma10": sma10, "sma50": sma50, "ema20": ema20, "ema50": ema50,
        "rsi": rsi_val, "macd": macd_val, "macd_signal": macd_sig, "macd_hist": macd_hist,
        "bb_up": bb_up, "bb_mid": bb_mid, "bb_low": bb_low, "atr": atr, "momentum": momentum
    }
    return {"signal": base, "confidence": round(conf,1), "entry": entry, "sl": sl, "tp": tp, "indicators": indicators}

# -------------------------
# Telegram bot handlers
# -------------------------
def start(update: Update, ctx: CallbackContext):
    chat_id = update.effective_chat.id
    session_state.setdefault(chat_id, {"pair":None,"timeframe":None,"account_choice":"auto","awaiting":None})
    msg = ("Welcome — Professional Forex Signal Bot.\n"
           "Use the keyboard to select pair/timeframe and request a signal.\n"
           "You can set your OANDA keys in Settings so live signals are real.")
    update.message.reply_text(msg, reply_markup=main_menu_kb())

def help_cmd(update: Update, ctx: CallbackContext):
    update.message.reply_text("Help:\n- Select Pair\n- Select Timeframe\n- Request Signal\n- Settings to provide OANDA keys\n- Subscribe to alerts", reply_markup=main_menu_kb())

def settings_cmd(update: Update, ctx: CallbackContext):
    chat_id = update.effective_chat.id
    session_state.setdefault(chat_id, {"pair":None,"timeframe":None,"account_choice":"auto","awaiting":None})
    update.message.reply_text("Settings: Choose account type to set keys or Back.", reply_markup=account_kb())

def handle_text(update: Update, ctx: CallbackContext):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    state = session_state.setdefault(chat_id, {"pair":None,"timeframe":None,"account_choice":"auto","awaiting":None})

    # If waiting for API key input
    if state.get("awaiting"):
        field = state["awaiting"]
        users.setdefault(str(chat_id), users.get(str(chat_id), {}))
        users[str(chat_id)][field] = text.strip()
        save_users(users)
        state["awaiting"] = None
        update.message.reply_text(f"{field} saved.", reply_markup=main_menu_kb())
        return

    # Main menu actions
    if text == "📈 Select Pair":
        update.message.reply_text("Choose pair:", reply_markup=pairs_kb()); return
    if text == "⏱ Select Timeframe":
        update.message.reply_text("Choose timeframe:", reply_markup=tfs_kb()); return
    if text == "⚙️ Settings":
        update.message.reply_text("Settings - choose account or set keys:", reply_markup=account_kb()); return
    if text == "ℹ️ Help":
        help_cmd(update, ctx); return
    if text == "🔔 Subscribe":
        subscribers.add(chat_id); update.message.reply_text("Subscribed to alerts.", reply_markup=main_menu_kb()); return
    if text == "🔕 Unsubscribe":
        subscribers.discard(chat_id); update.message.reply_text("Unsubscribed.", reply_markup=main_menu_kb()); return

    # Account keyboard choices
    if text in ("Practice","Live","Auto (Prefer Practice)"):
        if text == "Practice":
            state["account_choice"]="practice"
            update.message.reply_text("Account set to Practice. You can /set_practice to enter API key or use signals (simulated if key missing).", reply_markup=main_menu_kb())
            return
        if text == "Live":
            state["account_choice"]="live"
            update.message.reply_text("Account set to Live. You can /set_live to enter API key.", reply_markup=main_menu_kb())
            return
        if text.startswith("Auto"):
            state["account_choice"]="auto"
            update.message.reply_text("Account set to Auto (prefer practice if available).", reply_markup=main_menu_kb())
            return

    # Pairs and timeframes selection
    if text in ("EURUSD","GBPUSD","USDJPY","AUDUSD","XAUUSD","USDCAD"):
        state["pair"]=text; update.message.reply_text(f"Pair set: {text}", reply_markup=main_menu_kb()); return
    if text in ("1m","5m","15m","30m","1h"):
        state["timeframe"]=text; update.message.reply_text(f"Timeframe set: {text}", reply_markup=main_menu_kb()); return

    if text == "Back":
        update.message.reply_text("Back to menu.", reply_markup=main_menu_kb()); return

    if text == "🎯 Request Signal":
        # generate signal
        pair = state.get("pair"); tf = state.get("timeframe")
        if not pair or not tf:
            update.message.reply_text("Please select both pair and timeframe first.", reply_markup=main_menu_kb()); return
        update.message.reply_text("Fetching data and computing signal... ⏳")
        candles = get_candles_for_user(chat_id, pair, tf)
        if not candles or len(candles) < MIN_BARS_REQUIRED:
            update.message.reply_text("Insufficient real data and simulated data disabled. Use Settings to provide OANDA key or allow simulated data.", reply_markup=main_menu_kb()); return
        sig = professional_signal(candles)
        zones = detect_support_resistance(candles)
        pats = detect_candle_patterns(candles)
        # build message (Professional + indicators)
        lines = []
        lines.append(f"*{pair}*  _{tf}_")
        lines.append(f"*Signal*: *{sig['signal']}*    _Confidence: {sig['confidence']}%_")
        if sig.get("entry") is not None:
            lines.append(f"*Entry*: {sig['entry']}  *SL*: {sig['sl']}  *TP*: {sig['tp']}")
        lines.append("\n*Indicators*:")
        ind = sig['indicators']
        lines.append(f" SMA10:{safe_float(ind.get('sma10'))} SMA50:{safe_float(ind.get('sma50'))}")
        lines.append(f" EMA20:{safe_float(ind.get('ema20'))} EMA50:{safe_float(ind.get('ema50'))}")
        lines.append(f" RSI:{safe_float(ind.get('rsi'))}  ATR:{round(ind.get('atr') or 0,6)}")
        macd_txt = f"MACD:{safe_float(ind.get('macd'))} SIG:{safe_float(ind.get('macd_signal'))} HIST:{safe_float(ind.get('macd_hist'))}"
        lines.append(macd_txt)
        if pats:
            lines.append(f"*Patterns*: {', '.join(pats)}")
        # support/res
        res_z = ", ".join([f"{z[0]}-{z[1]}" for z in zones.get("resistance",[])])
        sup_z = ", ".join([f"{z[0]}-{z[1]}" for z in zones.get("support",[])])
        if res_z:
            lines.append(f"*Resistance zones*: {res_z}")
        if sup_z:
            lines.append(f"*Support zones*: {sup_z}")
        msg = "\n".join(lines)
        update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())
        # broadcast to subscribers if high confidence
        try:
            if sig['confidence'] >= 80:
                note = f"ALERT {pair} {tf} {sig['signal']} ({sig['confidence']}%)"
                broadcast_to_subscribers(note)
        except Exception:
            pass
        return

    if text == "🔎 Check Trend":
        pair = state.get("pair"); tf=state.get("timeframe")
        if not pair or not tf:
            update.message.reply_text("Select pair & timeframe first.", reply_markup=main_menu_kb()); return
        candles = get_candles_for_user(chat_id, pair, tf)
        if not candles:
            update.message.reply_text("No data available.", reply_markup=main_menu_kb()); return
        closes = [c['close'] for c in candles]
        s10 = sma(closes,10); s50 = sma(closes,50)
        trend = "Unknown"
        if s10 and s50:
            trend = "Bullish" if s10 > s50 else "Bearish"
        update.message.reply_text(f"Trend: {trend}\nSMA10: {safe_float(s10)}  SMA50: {safe_float(s50)}", reply_markup=main_menu_kb())
        return

    if text == "🧾 Support/Resistance":
        pair = state.get("pair"); tf=state.get("timeframe")
        if not pair or not tf:
            update.message.reply_text("Select pair & timeframe first.", reply_markup=main_menu_kb()); return
        candles = get_candles_for_user(chat_id, pair, tf)
        if not candles:
            update.message.reply_text("No data available.", reply_markup=main_menu_kb()); return
        zones = detect_support_resistance(candles)
        res_z = ", ".join([f"{z[0]}-{z[1]}" for z in zones.get("resistance",[])])
        sup_z = ", ".join([f"{z[0]}-{z[1]}" for z in zones.get("support",[])])
        msg = f"Resistance: {res_z}\nSupport: {sup_z}"
        update.message.reply_text(msg, reply_markup=main_menu_kb())
        return

    if text == "📰 Candles/Patterns":
        pair = state.get("pair"); tf=state.get("timeframe")
        if not pair or not tf:
            update.message.reply_text("Select pair & timeframe first.", reply_markup=main_menu_kb()); return
        candles = get_candles_for_user(chat_id, pair, tf)
        pats = detect_candle_patterns(candles) if candles else []
        update.message.reply_text(f"Recent patterns: {', '.join(pats) if pats else 'None'}", reply_markup=main_menu_kb())
        return

    # Settings: commands to set API keys
    if text == "/set_practice" or text.lower()=="set practice" or text=="Set Practice":
        state["awaiting"]="oanda_practice"
        update.message.reply_text("Send me your OANDA Practice API key now (paste the token).", reply_markup=main_menu_kb()); return
    if text == "/set_live" or text.lower()=="set live" or text=="Set Live":
        state["awaiting"]="oanda_live"
        update.message.reply_text("Send me your OANDA Live API key now (paste the token).", reply_markup=main_menu_kb()); return

    # fallback
    update.message.reply_text("Unknown command or button. Use the keyboard.", reply_markup=main_menu_kb())

def safe_float(x):
    try:
        return round(float(x),6)
    except:
        return None

def broadcast_to_subscribers(message_text:str):
    for cid in list(subscribers):
        try:
            updater.bot.send_message(chat_id=cid, text=message_text)
        except Exception:
            subscribers.discard(cid)

# -------------------------
# Command wrappers
# -------------------------
def cmd_start(update: Update, ctx: CallbackContext):
    start(update, ctx)

def cmd_help(update: Update, ctx: CallbackContext):
    help_cmd(update, ctx)

def cmd_set_practice(update: Update, ctx: CallbackContext):
    chat_id = update.effective_chat.id
    session_state.setdefault(chat_id, {"pair":None,"timeframe":None,"account_choice":"auto","awaiting":None})
    session_state[chat_id]["awaiting"]="oanda_practice"
    update.message.reply_text("Send your OANDA practice API key now (paste token).", reply_markup=main_menu_kb())

def cmd_set_live(update: Update, ctx: CallbackContext):
    chat_id = update.effective_chat.id
    session_state.setdefault(chat_id, {"pair":None,"timeframe":None,"account_choice":"auto","awaiting":None})
    session_state[chat_id]["awaiting"]="oanda_live"
    update.message.reply_text("Send your OANDA live API key now (paste token).", reply_markup=main_menu_kb())

# -------------------------
# Startup
# -------------------------
def main():
    if TELEGRAM_TOKEN.startswith("PASTE") or not TELEGRAM_TOKEN:
        print("Please set TELEGRAM_TOKEN in the script or environment variable TELEGRAM_TOKEN.")
        return
    updater.dispatcher.add_handler(CommandHandler("start", cmd_start))
    updater.dispatcher.add_handler(CommandHandler("help", cmd_help))
    updater.dispatcher.add_handler(CommandHandler("set_practice", cmd_set_practice))
    updater.dispatcher.add_handler(CommandHandler("set_live", cmd_set_live))
    updater.dispatcher.add_handler(MessageHandler(Filters.text & (~Filters.command), handle_text))
    updater.start_polling()
    print("Bot is running (polling). Press Ctrl-C to stop.")
    updater.idle()

if __name__ == "__main__":
    main()
