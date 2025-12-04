"""
app.py - Professional Telegram Forex Signal Bot (single-file)
- Reply keyboard interface
- Full technical indicators (SMA/EMA, RSI, MACD, Bollinger, ATR)
- Support/Resistance & candle pattern detection
- Professional signals with Entry/SL/TP + indicator details
- Users provide OANDA practice/live keys via Telegram
- Data sourced ONLY from OANDA API (no simulation)
"""

import os
import json
import math
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
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "") or "PASTE_YOUR_TELEGRAM_TOKEN_HERE"
USER_FILE = "users.json"
DEFAULT_BARS = 200
MIN_BARS_REQUIRED = 50

# OANDA endpoints
OANDA_PRACTICE_BASE = "https://api-fxpractice.oanda.com/v3/instruments/{instrument}/candles"
OANDA_LIVE_BASE = "https://api-fxtrade.oanda.com/v3/instruments/{instrument}/candles"

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("forex_bot")

# -------------------------
# Global variables
# -------------------------
subscribers = set()
updater = None
users = {}
session_state: Dict[int, Dict] = {}

# -------------------------
# Load / save users
# -------------------------
def load_users() -> Dict:
    if os.path.exists(USER_FILE):
        try:
            with open(USER_FILE, "r") as f:
                return json.load(f)
        except Exception:
            logger.error("Failed to load users file")
            return {}
    return {}

def save_users(data: Dict):
    try:
        with open(USER_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed saving users: {e}")

users = load_users()

# -------------------------
# Keyboards
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
    times = ["1m","5m","15m","30m","1h","4h","1d","Back"]
    rows = [times[i:i+3] for i in range(0,len(times),3)]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def account_kb():
    rows = [["Practice","Live","Auto (Prefer Practice)"],["Back"]]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

def settings_kb():
    rows = [
        ["Set Practice Key", "Set Live Key"],
        ["Check API Status", "Back"]
    ]
    return ReplyKeyboardMarkup(rows, resize_keyboard=True)

# -------------------------
# OANDA Data Fetching (REAL DATA ONLY)
# -------------------------
def fetch_oanda_candles(pair: str, timeframe: str, count: int = DEFAULT_BARS, 
                       account_type: str = "practice", api_key: str = None):
    """
    Fetch REAL candles from OANDA API only
    Returns: List of candles or None if failed
    """
    if not api_key:
        return None
    
    try:
        # Map timeframe to OANDA granularity
        gran_map = {
            "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
            "1h": "H1", "4h": "H4", "1d": "D", "1w": "W", "1mn": "M"
        }
        
        gran = gran_map.get(timeframe.lower())
        if not gran:
            logger.error(f"Unsupported timeframe: {timeframe}")
            return None
        
        instrument = normalize_pair(pair)
        base = OANDA_PRACTICE_BASE if account_type == "practice" else OANDA_LIVE_BASE
        url = base.format(instrument=instrument)
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        params = {
            "granularity": gran,
            "count": count,
            "price": "M"  # Mid prices
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=15)
        
        if response.status_code != 200:
            logger.error(f"OANDA API Error {response.status_code}: {response.text[:200]}")
            return None
        
        data = response.json()
        
        if "candles" not in data or not data["candles"]:
            logger.error("No candles in OANDA response")
            return None
        
        candles = []
        for candle in data["candles"]:
            if candle["complete"]:  # Only use complete candles
                candles.append({
                    "time": candle.get("time"),
                    "open": float(candle["mid"]["o"]),
                    "high": float(candle["mid"]["h"]),
                    "low": float(candle["mid"]["l"]),
                    "close": float(candle["mid"]["c"]),
                    "volume": int(candle.get("volume", 0))
                })
        
        # Reverse to get oldest first
        candles.reverse()
        
        if len(candles) < MIN_BARS_REQUIRED:
            logger.warning(f"Insufficient candles: {len(candles)}")
            return None
            
        return candles
        
    except requests.exceptions.Timeout:
        logger.error("OANDA API timeout")
        return None
    except Exception as e:
        logger.error(f"OANDA fetch error: {e}")
        return None

def normalize_pair(pair: str) -> str:
    """Format pair for OANDA API"""
    pair = pair.strip().upper().replace("/", "_")
    if "_" not in pair and len(pair) == 6:
        pair = f"{pair[:3]}_{pair[3:]}"
    return pair

def get_user_api_key(chat_id: int, account_type: str) -> Optional[str]:
    """Get user's API key for specified account type"""
    user_data = users.get(str(chat_id), {})
    return user_data.get(f"oanda_{account_type}")

def get_candles_for_user(chat_id: int, pair: str, timeframe: str, 
                        account_choice: str = "auto") -> Optional[List[Dict]]:
    """
    Get REAL candles for user based on their API keys
    Returns None if no API keys or API fails
    """
    user_data = users.get(str(chat_id), {})
    
    # Determine which API key to use
    if account_choice == "practice":
        api_key = user_data.get("oanda_practice")
        if api_key:
            candles = fetch_oanda_candles(pair, timeframe, account_type="practice", api_key=api_key)
            if candles:
                return candles
    
    elif account_choice == "live":
        api_key = user_data.get("oanda_live")
        if api_key:
            candles = fetch_oanda_candles(pair, timeframe, account_type="live", api_key=api_key)
            if candles:
                return candles
    
    else:  # auto - try practice first, then live
        api_key = user_data.get("oanda_practice")
        if api_key:
            candles = fetch_oanda_candles(pair, timeframe, account_type="practice", api_key=api_key)
            if candles:
                return candles
        
        api_key = user_data.get("oanda_live")
        if api_key:
            candles = fetch_oanda_candles(pair, timeframe, account_type="live", api_key=api_key)
            if candles:
                return candles
    
    # No API keys or all failed
    return None

# -------------------------
# Technical Indicators (same as before)
# -------------------------
def sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period

def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    ema_val = sum(values[:period]) / period
    for price in values[period:]:
        ema_val = (price - ema_val) * k + ema_val
    return ema_val

def rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) <= period:
        return None
    gains = []
    losses = []
    for i in range(1, period + 1):
        diff = values[-i] - values[-i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(-diff)
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period if sum(losses) != 0 else 1e-9
    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    return 100 - (100 / (1 + rs))

def stddev(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    subset = values[-period:]
    return statistics.pstdev(subset)

def bollinger(values: List[float], period: int = 20, devs: float = 2.0):
    m = sma(values, period)
    sd = stddev(values, period)
    if m is None or sd is None:
        return None, None, None
    return m + devs * sd, m, m - devs * sd

def macd(values: List[float], fast: int = 12, slow: int = 26, signal: int = 9):
    if len(values) < slow + signal:
        return None, None, None
    
    # Compute MACD value
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    if ema_fast is None or ema_slow is None:
        return None, None, None
    
    macd_val = ema_fast - ema_slow
    
    # Compute MACD series for signal line
    macd_series = []
    for i in range(slow, len(values) + 1):
        sub = values[:i]
        ef = ema(sub, fast)
        es = ema(sub, slow)
        if ef is not None and es is not None:
            macd_series.append(ef - es)
    
    if len(macd_series) < signal:
        return macd_val, None, None
    
    sig = ema(macd_series, signal)
    hist = macd_val - sig if sig is not None else None
    return macd_val, sig, hist

def atr(candles: List[Dict], period: int = 14) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    
    trs = []
    for i in range(1, len(candles)):
        high = candles[i]['high']
        low = candles[i]['low']
        prev_close = candles[i - 1]['close']
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    
    if len(trs) < period:
        return None
    
    return sum(trs[-period:]) / period

# -------------------------
# Support/Resistance
# -------------------------
def detect_support_resistance(candles: List[Dict], lookback: int = 100, zone_width: float = 0.0015):
    recent = candles[-lookback:] if len(candles) > lookback else candles
    
    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(recent) - 2):
        if (recent[i]['high'] > recent[i-1]['high'] and 
            recent[i]['high'] > recent[i-2]['high'] and
            recent[i]['high'] > recent[i+1]['high'] and
            recent[i]['high'] > recent[i+2]['high']):
            swing_highs.append((i, recent[i]['high']))
        
        if (recent[i]['low'] < recent[i-1]['low'] and 
            recent[i]['low'] < recent[i-2]['low'] and
            recent[i]['low'] < recent[i+1]['low'] and
            recent[i]['low'] < recent[i+2]['low']):
            swing_lows.append((i, recent[i]['low']))
    
    # Take top 3 most recent swings
    swing_highs.sort(key=lambda x: -x[1])
    swing_lows.sort(key=lambda x: x[1])
    
    zones = {"resistance": [], "support": []}
    
    for _, h in swing_highs[:3]:
        zones["resistance"].append((
            round(h * (1 - zone_width), 6),
            round(h * (1 + zone_width), 6)
        ))
    
    for _, l in swing_lows[:3]:
        zones["support"].append((
            round(l * (1 - zone_width), 6),
            round(l * (1 + zone_width), 6)
        ))
    
    return zones

# -------------------------
# Candlestick Patterns
# -------------------------
def detect_candle_patterns(candles: List[Dict]) -> List[str]:
    if len(candles) < 3:
        return []
    
    patterns = []
    last = candles[-1]
    prev = candles[-2]
    prev2 = candles[-3]
    
    # Bullish Engulfing
    if (prev['close'] < prev['open'] and 
        last['close'] > last['open'] and 
        last['close'] > prev['open'] and 
        last['open'] < prev['close']):
        patterns.append("Bullish Engulfing")
    
    # Bearish Engulfing
    if (prev['close'] > prev['open'] and 
        last['close'] < last['open'] and 
        last['open'] > prev['close'] and 
        last['close'] < prev['open']):
        patterns.append("Bearish Engulfing")
    
    # Morning Star (simplified)
    if (prev2['close'] < prev2['open'] and  # Downward trend
        abs(prev['close'] - prev['open']) < (prev2['high'] - prev2['low']) * 0.3 and  # Small body
        last['close'] > last['open'] and  # Up candle
        last['close'] > (prev2['open'] + prev2['close']) / 2):  # Closes above midpoint
        patterns.append("Morning Star")
    
    # Evening Star (simplified)
    if (prev2['close'] > prev2['open'] and  # Upward trend
        abs(prev['close'] - prev['open']) < (prev2['high'] - prev2['low']) * 0.3 and  # Small body
        last['close'] < last['open'] and  # Down candle
        last['close'] < (prev2['open'] + prev2['close']) / 2):  # Closes below midpoint
        patterns.append("Evening Star")
    
    return patterns

# -------------------------
# Signal Engine (REAL DATA ONLY)
# -------------------------
def generate_signal(candles: List[Dict]) -> Dict:
    """Generate trading signal based on REAL market data"""
    if len(candles) < 50:
        return {
            "signal": "HOLD",
            "confidence": 0,
            "reason": "Insufficient data",
            "entry": None,
            "sl": None,
            "tp": None,
            "indicators": {}
        }
    
    closes = [c['close'] for c in candles]
    current_price = closes[-1]
    
    # Calculate all indicators
    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    rsi_val = rsi(closes, 14)
    macd_val, macd_sig, macd_hist = macd(closes)
    bb_upper, bb_middle, bb_lower = bollinger(closes, 20, 2.0)
    atr_val = atr(candles, 14)
    
    # Signal scoring
    buy_score = 0
    sell_score = 0
    reasons = []
    
    # Trend Analysis
    if sma20 and sma50:
        if sma20 > sma50:
            buy_score += 2
            reasons.append("SMA20 > SMA50 (Bullish)")
        else:
            sell_score += 2
            reasons.append("SMA20 < SMA50 (Bearish)")
    
    if ema12 and ema26:
        if ema12 > ema26:
            buy_score += 1
            reasons.append("EMA12 > EMA26 (Bullish)")
        else:
            sell_score += 1
            reasons.append("EMA12 < EMA26 (Bearish)")
    
    # Momentum
    if len(closes) >= 5:
        momentum = closes[-1] - closes[-5]
        if momentum > 0:
            buy_score += 1
            reasons.append("Positive 5-period momentum")
        else:
            sell_score += 1
            reasons.append("Negative 5-period momentum")
    
    # RSI Analysis
    if rsi_val:
        if rsi_val < 30:
            buy_score += 2
            reasons.append("RSI Oversold (<30)")
        elif rsi_val > 70:
            sell_score += 2
            reasons.append("RSI Overbought (>70)")
        elif 30 <= rsi_val <= 70:
            buy_score += 0.5 if rsi_val > 50 else 0
            sell_score += 0.5 if rsi_val < 50 else 0
    
    # MACD Analysis
    if macd_val and macd_sig:
        if macd_val > macd_sig:
            buy_score += 1
            reasons.append("MACD > Signal (Bullish)")
        else:
            sell_score += 1
            reasons.append("MACD < Signal (Bearish)")
    
    # Bollinger Bands Position
    if bb_lower and bb_upper:
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
        
        if bb_position < 0.2:
            buy_score += 1
            reasons.append("Price near lower BB (Oversold)")
        elif bb_position > 0.8:
            sell_score += 1
            reasons.append("Price near upper BB (Overbought)")
    
    # Determine Signal
    signal = "HOLD"
    confidence = 0
    
    if buy_score > sell_score + 2:  # Strong buy
        signal = "BUY"
        confidence = min(95, 50 + (buy_score - sell_score) * 5)
    elif sell_score > buy_score + 2:  # Strong sell
        signal = "SELL"
        confidence = min(95, 50 + (sell_score - buy_score) * 5)
    else:
        signal = "HOLD"
        confidence = max(0, 50 - abs(buy_score - sell_score) * 5)
    
    # Calculate Risk/Reward
    entry = round(current_price, 6)
    stop_loss = None
    take_profit = None
    
    if atr_val and signal in ["BUY", "SELL"]:
        if signal == "BUY":
            stop_loss = round(entry - atr_val * 1.5, 6)
            take_profit = round(entry + atr_val * 3, 6)
        elif signal == "SELL":
            stop_loss = round(entry + atr_val * 1.5, 6)
            take_profit = round(entry - atr_val * 3, 6)
    
    # Prepare indicators dictionary
    indicators = {
        "sma20": sma20,
        "sma50": sma50,
        "ema12": ema12,
        "ema26": ema26,
        "rsi": rsi_val,
        "macd": macd_val,
        "macd_signal": macd_sig,
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "atr": atr_val,
        "current_price": current_price
    }
    
    return {
        "signal": signal,
        "confidence": round(confidence, 1),
        "reason": ", ".join(reasons) if reasons else "No clear trend",
        "entry": entry,
        "sl": stop_loss,
        "tp": take_profit,
        "indicators": indicators
    }

# -------------------------
# Telegram Bot Handlers
# -------------------------
def start(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    session_state[chat_id] = {
        "pair": "EURUSD",
        "timeframe": "1h",
        "account_choice": "auto",
        "awaiting": None
    }
    
    welcome_msg = """🤖 *Forex Signal Bot*
    
*REAL MARKET DATA ONLY*
This bot uses ONLY real OANDA API data - no simulations or random signals.

*Before using:*
1. Get OANDA API keys (practice or live)
2. Set your API keys via Settings
3. Select pair and timeframe
4. Request signals

⚠️ *Warning:* Trading involves risk. Use signals at your own discretion."""
    
    update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

def help_command(update: Update, context: CallbackContext):
    help_text = """*Available Commands:*
    
*Main Functions:*
📈 Select Pair - Choose currency pair
⏱ Select Timeframe - Select chart timeframe
🎯 Request Signal - Generate trading signal
🔎 Check Trend - Analyze market trend
🧾 Support/Resistance - Show key levels
📰 Candles/Patterns - Detect candlestick patterns

*Settings:*
⚙️ Settings - Configure API keys
🔔 Subscribe - Get signal alerts
🔕 Unsubscribe - Stop alerts

*API Setup:*
1. Get OANDA API key from your account
2. Practice account: https://www.oanda.com/practice-account/
3. Use /set_practice or /set_live to save keys

*Note:* All signals are based on REAL OANDA market data."""
    
    update.message.reply_text(help_text, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())

def handle_text(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    text = update.message.text.strip()
    
    # Initialize user session
    if chat_id not in session_state:
        session_state[chat_id] = {
            "pair": "EURUSD",
            "timeframe": "1h",
            "account_choice": "auto",
            "awaiting": None
        }
    
    state = session_state[chat_id]
    
    # Handle API key input
    if state.get("awaiting"):
        field = state["awaiting"]
        
        if len(text) < 20:  # Basic validation
            update.message.reply_text("Invalid API key format. Key should be longer.", reply_markup=main_menu_kb())
            state["awaiting"] = None
            return
        
        # Save API key
        if str(chat_id) not in users:
            users[str(chat_id)] = {}
        
        users[str(chat_id)][field] = text.strip()
        save_users(users)
        
        update.message.reply_text(f"✅ {field.replace('_', ' ').title()} saved successfully!", reply_markup=main_menu_kb())
        state["awaiting"] = None
        return
    
    # Main menu navigation
    if text == "📈 Select Pair":
        update.message.reply_text("Select currency pair:", reply_markup=pairs_kb())
        return
    
    elif text == "⏱ Select Timeframe":
        update.message.reply_text("Select timeframe:", reply_markup=tfs_kb())
        return
    
    elif text == "🎯 Request Signal":
        pair = state.get("pair")
        timeframe = state.get("timeframe")
        
        if not pair or not timeframe:
            update.message.reply_text("Please select pair and timeframe first.", reply_markup=main_menu_kb())
            return
        
        # Check if user has API keys
        user_keys = users.get(str(chat_id), {})
        has_keys = "oanda_practice" in user_keys or "oanda_live" in user_keys
        
        if not has_keys:
            update.message.reply_text(
                "❌ *No API keys found!*\n\n"
                "Please set your OANDA API keys first:\n"
                "1. Go to Settings\n"
                "2. Choose 'Set Practice Key' or 'Set Live Key'\n"
                "3. Paste your OANDA API token\n\n"
                "Get API keys from:\n"
                "• Practice: https://www.oanda.com/practice-account/\n"
                "• Live: Your OANDA account dashboard",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_menu_kb()
            )
            return
        
        update.message.reply_text(f"⏳ Fetching {pair} {timeframe} data from OANDA...", reply_markup=main_menu_kb())
        
        # Get REAL data from OANDA
        candles = get_candles_for_user(chat_id, pair, timeframe, state.get("account_choice", "auto"))
        
        if not candles:
            update.message.reply_text(
                "❌ *Failed to fetch market data!*\n\n"
                "Possible reasons:\n"
                "• Invalid API key\n"
                "• API rate limit exceeded\n"
                "• Network issue\n"
                "• OANDA server down\n\n"
                "Check your API keys in Settings.",
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=main_menu_kb()
            )
            return
        
        # Generate signal
        signal = generate_signal(candles)
        patterns = detect_candle_patterns(candles)
        zones = detect_support_resistance(candles)
        
        # Build signal message
        message_lines = [
            f"*{pair}*  _{timeframe}_",
            f"*Signal*: *{signal['signal']}*  _(Confidence: {signal['confidence']}%)_",
            f"*Reason*: {signal['reason']}",
            ""
        ]
        
        if signal['entry'] and signal['sl'] and signal['tp']:
            message_lines.append("*Trade Setup:*")
            message_lines.append(f"Entry: `{signal['entry']}`")
            message_lines.append(f"Stop Loss: `{signal['sl']}`")
            message_lines.append(f"Take Profit: `{signal['tp']}`")
            if signal['tp'] and signal['entry']:
                rr_ratio = abs((signal['tp'] - signal['entry']) / (signal['entry'] - signal['sl'])) if signal['entry'] != signal['sl'] else 0
                message_lines.append(f"Risk/Reward: 1:{round(rr_ratio, 2)}")
            message_lines.append("")
        
        message_lines.append("*Technical Indicators:*")
        ind = signal['indicators']
        message_lines.append(f"SMA20: `{safe_float(ind.get('sma20'))}`  SMA50: `{safe_float(ind.get('sma50'))}`")
        message_lines.append(f"EMA12: `{safe_float(ind.get('ema12'))}`  EMA26: `{safe_float(ind.get('ema26'))}`")
        message_lines.append(f"RSI: `{safe_float(ind.get('rsi'))}`  ATR: `{safe_float(ind.get('atr'))}`")
        message_lines.append(f"MACD: `{safe_float(ind.get('macd'))}`  Signal: `{safe_float(ind.get('macd_signal'))}`")
        
        if patterns:
            message_lines.append(f"\n*Candle Patterns:* {', '.join(patterns)}")
        
        # Add support/resistance
        if zones.get('support') or zones.get('resistance'):
            message_lines.append("\n*Key Levels:*")
            for sup in zones.get('support', [])[:2]:
                message_lines.append(f"Support: `{sup[0]} - {sup[1]}`")
            for res in zones.get('resistance', [])[:2]:
                message_lines.append(f"Resistance: `{res[0]} - {res[1]}`")
        
        message_lines.append("\n⚠️ *Risk Warning:* Trading involves risk. Past performance doesn't guarantee future results.")
        
        update.message.reply_text(
            "\n".join(message_lines),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=main_menu_kb()
        )
        
        # Broadcast to subscribers for strong signals
        if signal['confidence'] >= 75:
            alert_msg = f"🚨 {pair} {timeframe}: {signal['signal']} ({signal['confidence']}%)"
            broadcast_to_subscribers(alert_msg)
        
        return
    
    elif text == "🔎 Check Trend":
        pair = state.get("pair")
        timeframe = state.get("timeframe")
        
        if not pair or not timeframe:
            update.message.reply_text("Please select pair and timeframe first.", reply_markup=main_menu_kb())
            return
        
        candles = get_candles_for_user(chat_id, pair, timeframe, state.get("account_choice", "auto"))
        
        if not candles:
            update.message.reply_text("Failed to fetch data. Check your API keys.", reply_markup=main_menu_kb())
            return
        
        closes = [c['close'] for c in candles]
        
        # Calculate trends
        sma20 = sma(closes, 20)
        sma50 = sma(closes, 50)
        rsi_val = rsi(closes, 14)
        
        trend = "Sideways"
        if sma20 and sma50:
            if sma20 > sma50:
                trend = "Bullish"
            else:
                trend = "Bearish"
        
        strength = "Weak"
        if rsi_val:
            if rsi_val > 60 or rsi_val < 40:
                strength = "Strong"
        
        message = f"""*Trend Analysis - {pair} {timeframe}*
        
*Primary Trend:* {trend}
*Trend Strength:* {strength}
*RSI:* {safe_float(rsi_val)}
*SMA20:* {safe_float(sma20)}
*SMA50:* {safe_float(sma50)}

*Price Action:*
Current: `{closes[-1] if closes else 'N/A'}`
24-bar High: `{max(closes[-24:]) if len(closes) >= 24 else 'N/A'}`
24-bar Low: `{min(closes[-24:]) if len(closes) >= 24 else 'N/A'}`"""
        
        update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())
        return
    
    elif text == "🧾 Support/Resistance":
        pair = state.get("pair")
        timeframe = state.get("timeframe")
        
        if not pair or not timeframe:
            update.message.reply_text("Please select pair and timeframe first.", reply_markup=main_menu_kb())
            return
        
        candles = get_candles_for_user(chat_id, pair, timeframe, state.get("account_choice", "auto"))
        
        if not candles:
            update.message.reply_text("Failed to fetch data. Check your API keys.", reply_markup=main_menu_kb())
            return
        
        zones = detect_support_resistance(candles)
        
        message_lines = [f"*Support/Resistance - {pair} {timeframe}*"]
        
        if zones['support']:
            message_lines.append("\n*Support Zones:*")
            for i, (low, high) in enumerate(zones['support'][:3], 1):
                message_lines.append(f"{i}. `{low} - {high}`")
        
        if zones['resistance']:
            message_lines.append("\n*Resistance Zones:*")
            for i, (low, high) in enumerate(zones['resistance'][:3], 1):
                message_lines.append(f"{i}. `{low} - {high}`")
        
        if not zones['support'] and not zones['resistance']:
            message_lines.append("\nNo clear support/resistance levels detected.")
        
        update.message.reply_text("\n".join(message_lines), parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())
        return
    
    elif text == "📰 Candles/Patterns":
        pair = state.get("pair")
        timeframe = state.get("timeframe")
        
        if not pair or not timeframe:
            update.message.reply_text("Please select pair and timeframe first.", reply_markup=main_menu_kb())
            return
        
        candles = get_candles_for_user(chat_id, pair, timeframe, state.get("account_choice", "auto"))
        
        if not candles:
            update.message.reply_text("Failed to fetch data. Check your API keys.", reply_markup=main_menu_kb())
            return
        
        patterns = detect_candle_patterns(candles)
        last_candle = candles[-1]
        
        candle_type = "Bullish" if last_candle['close'] > last_candle['open'] else "Bearish"
        body_size = abs(last_candle['close'] - last_candle['open'])
        total_range = last_candle['high'] - last_candle['low']
        body_ratio = (body_size / total_range * 100) if total_range > 0 else 0
        
        message = f"""*Candlestick Analysis - {pair} {timeframe}*

*Last Candle:*
Type: {candle_type}
Open: `{last_candle['open']}`
High: `{last_candle['high']}`
Low: `{last_candle['low']}`
Close: `{last_candle['close']}`
Body: {round(body_ratio, 1)}% of range

*Detected Patterns:*
{', '.join(patterns) if patterns else 'No significant patterns'}

*Candle Interpretation:*
{'Strong momentum' if body_ratio > 70 else 'Moderate momentum' if body_ratio > 30 else 'Indecision'}"""
        
        update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN, reply_markup=main_menu_kb())
        return
    
    elif text == "⚙️ Settings":
        update.message.reply_text("Settings:", reply_markup=settings_kb())
        return
    
    elif text == "ℹ️ Help":
        help_command(update, context)
        return
    
    elif text == "🔔 Subscribe":
        subscribers.add(chat_id)
        update.message.reply_text("✅ Subscribed to signal alerts!", reply_markup=main_menu_kb())
        return
    
    elif text == "🔕 Unsubscribe":
        subscribers.discard(chat_id)
        update.message.reply_text("✅ Unsubscribed from alerts.", reply_markup=main_menu_kb())
        return
    
    elif text == "Set Practice Key":
        state["awaiting"] = "oanda_practice"
        update.message.reply_text(
            "📝 *Setting Practice API Key*\n\n"
            "1. Go to https://www.oanda.com/practice-account/\n"
            "2. Create/l