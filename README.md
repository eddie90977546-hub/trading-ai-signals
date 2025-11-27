# signals.py
# Daily multi-asset long-only trading signals with risk sizing and drawdown control.
# Assets: BTC, Gold, Silver, Forex majors, and a broad US stock list.
# Dependencies: yfinance, pandas, numpy, scikit-learn, ta

import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# =========================
# Configuration
# =========================
ASSETS_CRYPTO = ["BTC-USD"]
ASSETS_COMMOD = ["XAUUSD=X", "XAGUSD=X"]
ASSETS_FX = ["EURUSD=X","GBPUSD=X","USDJPY=X","USDCAD=X","AUDUSD=X","NZDUSD=X"]
ASSETS_STOCKS = [
    "SPY","QQQ","IWM",
    "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","JPM","XOM","UNH","V",
    "XLF","XLE","XLK"
]
ASSETS = ASSETS_CRYPTO + ASSETS_COMMOD + ASSETS_FX + ASSETS_STOCKS

START_DATE = "2015-01-01"

# Risk parameters
RISK_PER_TRADE = 0.008     # 0.8% of equity
MAX_PER_ASSET = 0.10       # 10% of equity cap per position
STOP_ATR_MULT = 1.2
HOLD_DAYS = 5

# Signal thresholds
MIN_ATR_PCT = 0.003        # 0.3% min daily movement
MIN_PROB_UP = 0.60         # ML confidence threshold

# Drawdown control
DD_PAUSE_THRESHOLD = 0.10  # Pause new entries at 10% drawdown
DD_RESUME_GAP = 0.05       # Resume when within 5% of peak

# =========================
# Data and features
# =========================
def fetch_daily(ticker: str, start: str = START_DATE) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    return df.rename(columns=str.lower)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Returns
        # Short, medium, and swing horizons
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_20"] = df["close"].pct_change(20)
    # Trend
    ema20 = EMAIndicator(df["close"], window=20).ema_indicator()
    ema50 = EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema20"] = ema20
    df["ema50"] = ema50
    df["ema_slope"] = ema20.diff()
    # MACD histogram
    macd = MACD(df["close"])
    df["macd_hist"] = macd.macd_diff()
    # RSI
    df["rsi"] = RSIIndicator(df["close"], window=14).rsi()
    # ATR and ATR%
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df["atr"] = atr
    df["atr_pct"] = atr / df["close"]
    # Target: next-day direction (for ML)
    df["target_up"] = (df["close"].shift(-1) > df["close"]).astype(int)
    return df.dropna()

# =========================
# Modeling
# =========================
FEATURES = ["ret_1","ret_5","ret_20","ema20","ema50","ema_slope","macd_hist","rsi","atr_pct"]

def train_ml(df: pd.DataFrame):
    X = df[FEATURES].values
    y = df["target_up"].values
    split = int(len(df) * 0.7)
    scaler = StandardScaler()
    X_train, y_train = X[:split], y[:split]
    X_test = X[split:]
    scaler.fit(X_train)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42
    )
    model.fit(scaler.transform(X_train), y_train)
    prob = model.predict_proba(scaler.transform(X_test))[:, 1]
    out = df.iloc[split:].copy()
    out["prob_up"] = prob
    return model, scaler, out

# =========================
# Signals and risk
# =========================
def rule_long(row: pd.Series) -> int:
    trend = (row["ema20"] > row["ema50"]) and (row["macd_hist"] > 0)
    rsi_ok = row["rsi"] < 70
    vol_ok = row["atr_pct"] > MIN_ATR_PCT
    return int(trend and rsi_ok and vol_ok)

def generate_signal(ticker: str):
    df = fetch_daily(ticker)
    if df is None or len(df) < 200:
        return None
    df = add_features(df)
    model, scaler, out = train_ml(df)
    last = out.iloc[-1].copy()
    rule = rule_long(last)
    ml_ok = int(last["prob_up"] > MIN_PROB_UP)
    long_signal = int(rule == 1 and ml_ok == 1)
    atr_stop = float(last["atr"] * STOP_ATR_MULT)

    return {
        "asset": ticker,
        "date": out.index[-1].strftime("%Y-%m-%d"),
        "close": float(df["close"].iloc[-1]),
        "prob_up": float(last["prob_up"]),
        "atr_pct": float(last["atr_pct"]),
        "rsi": float(last["rsi"]),
        "long_signal": long_signal,
        "stop_distance": atr_stop,
        "hold_days": HOLD_DAYS,
        "rationale": "EMA20>EMA50 & MACD+ & RSI<70 & ATR%>0.3% & ML prob>0.60"
    }

class EquityCurve:
    def __init__(self, start_equity: float = 1.0):
        self.equity = start_equity
        self.peak = start_equity
        self.paused = False

    def update(self, daily_return: float):
        self.equity *= (1 + daily_return)
        self.peak = max(self.peak, self.equity)
        dd = (self.peak - self.equity) / self.peak if self.peak > 0 else 0.0
        if dd >= DD_PAUSE_THRESHOLD:
            self.paused = True
        elif self.paused and dd <= DD_RESUME_GAP:
            self.paused = False
        return dd, self.paused

def position_size(equity: float, price: float, atr: float) -> float:
    risk_dollars = equity * RISK_PER_TRADE
    stop_dollars = atr * STOP_ATR_MULT
    if stop_dollars <= 0 or price <= 0:
        return 0.0
    qty = risk_dollars / stop_dollars
    value = qty * price
    cap = equity * MAX_PER_ASSET
    if value > cap:
        qty = cap / price
    return float(max(qty, 0.0))

# =========================
# Main
# =========================
def main():
    print(f"Running daily signals for {len(ASSETS)} assets on {datetime.utcnow().strftime('%Y-%m-%d')}...")
    signals = []
    for t in ASSETS:
        try:
            s = generate_signal(t)
            if s:
                signals.append(s)
        except Exception as e:
            # Skip assets with data or indicator issues
            print(f"Warning: {t} skipped ({e})")
            continue

    if not signals:
        print("No signals generated.")
        return

    df_signals = pd.DataFrame(signals).sort_values(by="prob_up", ascending=False)
    # Example equity and sizing preview (not executing trades here)
    example_equity = 10000.0  # USD
    df_signals["position_qty"] = df_signals.apply(
        lambda r: position_size(example_equity, r["close"], r["stop_distance"] / STOP_ATR_MULT),
        axis=1
    )

    print("\nDaily long-only signals:")
    cols = ["asset","date","long_signal","prob_up","atr_pct","rsi","stop_distance","hold_days","position_qty","rationale"]
    print(df_signals[cols].to_string(index=False))

if __name__ == "__main__":
    main()# trading-ai-signals
For receiving ai trading signal 
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
ta>=0.10.2
# Daily Multi-Asset Trading Signals (Long-Only)

This project generates daily long-only trade signals for:
- Crypto: BTC-USD
- Commodities: Gold (XAUUSD=X), Silver (XAGUSD=X)
- Forex: EURUSD=X, GBPUSD=X, USDJPY=X, USDCAD=X, AUDUSD=X, NZDUSD=X
- Stocks/ETFs: SPY, QQQ, IWM, AAPL, MSFT, AMZN, NVDA, GOOGL, META, TSLA, JPM, XOM, UNH, V, XLF, XLE, XLK

## Features
- Technical indicators: EMA(20), EMA(50), MACD histogram, RSI(14), ATR/ATR%
- Random Forest classifier for next-day direction
- ATR-based stop distance and position sizing
- Portfolio drawdown monitor (pause new entries at 10%, resume within 5% of peak)

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
