"""
app.py - Telegram Forex Signal Bot (Single-file, Polling)
- Primary data: OANDA (practice & live)
- Polling (no webhook) - works in Acode/ArcCode
- Minimal deps: python-telegram-bot==13.15, requests
"""

import os
import logging
import random
import math
import statistics
import requests
from datetime import datetime, timedelta
from typing import List, Dict

from telegram import Bot, Update, ReplyKeyboardMarkup, KeyboardButton, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

# ----------------------------
# CONFIG - put your keys here or set environment variables
# ----------------------------
# Your Telegram bot token (you provided earlier). For safety, consider replacing token in BotFather after testing.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "") or "8346561514:AAHrcOAq_LkQvpADMWgt2bW3wrd-CrfQOXQ"

# OANDA practice & live API keys (replace with your keys or export as env vars)
OANDA_PRACTICE_API_KEY = os.getenv("OANDA_PRACTICE_API_KEY", "")  # paste if available
OANDA_LIVE_API_KEY = os.getenv("OANDA_LIVE_API_KEY", "")        # paste if available

# TwelveData API (optional fallback)
TWELVE_API_KEY = os.getenv("TWELVE_API_KEY", "")

# Settings
DEFAULT_BARS = 200
MIN_BARS_REQUIRED = 50
USE_REAL_DATA_ONLY = True  # if True, will not return simulated data unless OANDA/12Data available

# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forex_bot")

# ----------------------------
# Bot + state
# ----------------------------
bot = Bot(token=TELEGRAM_TOKEN)
updater = Updater(bot=bot, use_context=True)
dispatcher = updater.dispatcher

# user state: chat_id -> {pair, timeframe, account}
user_state: Dict[int, Dict] = {}
subscribers = set()  # chat ids to send instant alerts

# ----------------------------
# Keyboards
# ----------------------------
def main_menu_kb():
    kb = [
        [KeyboardButton("📈 Select Pair"), KeyboardButton("⏱ Select Timeframe")],
        [KeyboardButton("🎯 Request Signal"), KeyboardButton("🔎 Check Trend")],
        [KeyboardButton("🔔 Subscribe"), KeyboardButton("🔕 Unsubscribe")],
        [KeyboardButton("ℹ️ Help")]
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=False)

def pairs_kb():
    kb = [
        [KeyboardButton("EURUSD"), KeyboardButton("GBPUSD"), KeyboardButton("USDJPY")],
        [KeyboardButton("AUDUSD"), KeyboardButton("XAUUSD"), KeyboardButton("USDCAD")],
        [KeyboardButton("Back")]
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=False)

def tfs_kb():
    kb = [
        [KeyboardButton("1m"), KeyboardButton("5m"), KeyboardButton("15m")],
        [KeyboardButton("30m"), KeyboardButton("1h"), KeyboardButton("Back")]
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=False)

def account_kb():
    kb = [
        [KeyboardButton("Practice"), KeyboardButton("Live"), KeyboardButton("Auto (Prefer Practice)")],
        [KeyboardButton("Back")]
    ]
    return ReplyKeyboardMarkup(kb, resize_keyboard=True, one_time_keyboard=False)

# ----------------------------
# Utils: OANDA instrument formatting
# ----------------------------
def oanda_instrument(pair: str) -> str:
    s = pair.strip().upper().replace("/", "").replace("_", "")
    if len(s) == 6:
        return s[:3] + "_" + s[3:]
    return pair

# ----------------------------
# Fetch candles: OANDA -> TwelveData -> simulated
# ----------------------------
def fetch_oanda(pair: str, timeframe: str, count: int = DEFAULT_BARS, account_type: str = "practice"):
    """
    pair: 'EURUSD' or 'EUR_USD' or 'EUR/USD'
    timeframe: '1m','5m','15m','30m','1h' -> map to OANDA granularity (M1, M5, M15, M30, H1)
    account_type: 'practice' or 'live'
    """
    try:
        gran_map = {"1m":"M1","5m":"M5","15m":"M15","30m":"M30","1h":"H1"}
        g = gran_map.get(timeframe.lower(), "M5")
        inst = oanda_instrument(pair)
        headers = {}
        if account_type == "live":
            key = OANDA_LIVE_API_KEY
            url = f"https://api-fxtrade.oanda.com/v3/instruments/{inst}/candles"
        else:
            key = OANDA_PRACTICE_API_KEY
            url = f"https://api-fxpractice.oanda.com/v3/instruments/{inst}/candles"
        if not key:
            logger.info("OANDA key for %s not provided", account_type)
            return None
        headers["Authorization"] = f"Bearer {key}"
        params = {"granularity": g, "count": count, "price": "M"}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            logger.warning("OANDA HTTP %s: %s", r.status_code, r.text[:200])
            return None
        data = r.json()
        if "candles" not in data:
            return None
        out = []
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
    except Exception as e:
        logger.exception("fetch_oanda failed: %s", e)
        return None

def fetch_twelvedata(pair: str, interval: str, outputsize: int = DEFAULT_BARS):
    try:
        if not TWELVE_API_KEY:
            return None
        symbol = pair.replace("/","").replace("_","")
        url = "https://api.twelvedata.com/time_series"
        params = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "apikey": TWELVE_API_KEY}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "values" not in data:
            return None
        vals = list(reversed(data["values"]))
        out=[]
        for v in vals:
            out.append({
                "time": v.get("datetime") or v.get("timestamp"),
                "open": float(v["open"]),
                "high": float(v["high"]),
                "low": float(v["low"]),
                "close": float(v["close"]),
                "volume": float(v.get("volume",0))
            })
        return out
    except Exception:
        logger.exception("twelvedata error")
        return None

def simulated_candles(base_price=1.1000, bars=DEFAULT_BARS):
    price = base_price
    out=[]
    for i in range(bars):
        change = random.normalvariate(0, 0.0006)
        open_p = price
        close = price + change
        high = max(open_p, close) + abs(random.normalvariate(0, 0.0002))
        low = min(open_p, close) - abs(random.normalvariate(0, 0.0002))
        out.append({"time": None,"open": round(open_p,6),"high": round(high,6),"low": round(low,6),"close": round(close,6),"volume": int(1000+random.random()*1000)})
        price = close
    return out

def get_candles(pair: str, timeframe: str, account_choice: str = "auto"):
    # account_choice: 'practice','live','auto'
    # try OANDA practice first if auto, else live when requested
    if account_choice == "live":
        data = fetch_oanda(pair, timeframe
