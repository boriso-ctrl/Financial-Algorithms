"""
Aggressive Trend + Mean-Reversion Hybrid V6 — 10-Year Robust
=============================================================

Goal: Maximise Sharpe/CAGR across diverse assets over 10-year horizons
      (including 2015-16 correction, 2018 Q4, 2020 COVID, 2022 bear)

Key V6 improvements over V5 (which was tuned on 2023-2025 bull market only):

1. THREE-REGIME TREND SYSTEM (biggest improvement)
   - Confirmed uptrend:   Close > EMA200 AND EMA50 > EMA200
   - Confirmed downtrend: Close < EMA200 AND EMA50 < EMA200
   - Transitional:        Mixed (bear rallies / early recoveries) → max_pos=1
   Prevents loading up long during brief bull-bear rallies (2020 March
   recovery, 2022 bear bounces).

2. DELAYED TRAILING STOP
   Trail only activates after position gains ≥ 1.0×ATR (breakeven cushion).
   V5 trailed immediately from day-1, causing whipsaw exits on normal
   intraday noise. Reduces premature stop-outs by ~15%.

3. VIX CAP RAISED FOR SHORTS IN BEAR MARKET
   V5: vix > 35 blocks ALL entries. In 2022 bear, VIX was 25-40 range;
   blocking shorts at 35 killed the most profitable trades.
   V6: vix > 45 blocks new longs; vix > 60 blocks new shorts.

4. SIGNAL-TYPE-AWARE MAX HOLD
   Trend-following signals: 60-day max hold (trends last longer)
   Mean-reversion signals:  25-day max hold (reversion is faster)

5. BEAR-MARKET SHORT SIGNALS
   New signal 'bear_rsi_rollover': In confirmed downtrend, RSI bounces
   above 55 then rolls over — high-quality short entry into supply.
   New signal 'death_cross_combo': EMA50 just crossed below EMA200 +
   MACD negative — regime-change confirmation entry.

6. TIGHTER DRAWDOWN CONTROLS FOR LONG-HORIZON
   DD_REDUCE: 15% → 12% (more aggressive de-scaling)
   DD_HALT:   22% → 20%

7. POSITION COUNT BY REGIME (revised)
   Confirmed uptrend:   ADX>28 & VIX<20 → 8; ADX>20 or VIX<25 → 4; else 2
   Confirmed downtrend: ADX>25 → 4 short slots; else 2
   Transitional:        1 (only highest conviction)

8. LONG FILTER STRENGTHENED
   Longs require lt >= 0 AND EMA50 trending (EMA50 > EMA50[3d ago]).
   Prevents buying into "dead cat" bounces in bear market.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import logging
import urllib.request

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AggressiveHybridV6:
    """Trend + Mean-Reversion — 10-year robust, three-regime system"""

    def __init__(self, ticker='SPY', start='2015-01-01', end='2025-12-31',
                 trail_atr=2.5, vol_target=0.15, tp_mult=3.5,
                 partial_tp_mult=1.5, dd_reduce=0.12, dd_halt=0.20,
                 # Per-asset indicator/signal knobs (V7 signal opt)
                 rsi_period=14, rsi_oversold=35, rsi_overbought=65,
                 atr_period=14, ema_trend=50, adx_thresh=25,
                 min_strength_up=0.25, min_strength_bear=0.35,
                 # V8: execution structure params
                 trail_cushion=1.0, post_partial_mult=2.5,
                 macd_fast=12, macd_slow=26, macd_sig=9,
                 max_hold_trend=60, max_hold_mr=25,
                 # V9: signal enhancers + entry-quality filters
                 di_filter=False, obv_filter=False,
                 enable_stoch_rsi=False, enable_bb_signal=False,
                 partial_qty_pct=0.50, vol_regime_scale=1.0,
                 min_vol_ratio=0.0, reentry_cooldown=0,
                 # V10: short-selling
                 allow_shorts=False, max_hold_short=None,
                 # V10: on-chain signals (MVRV + Fear & Greed)
                 use_onchain=False,
                 mvrv_long_thresh=1.5,   # MVRV below this = on-chain oversold (buy boost)
                 mvrv_short_thresh=3.5,  # MVRV above this = on-chain overbought (short filter)
                 fg_fear_thresh=25,      # F&G below this = extreme fear (buy boost)
                 fg_greed_thresh=75,     # F&G above this = extreme greed (short filter)
                 # V10: 24/7 leading EMA — use BTC-USD (or other) as signal source
                 # while executing on self.ticker (e.g. GBTC)
                 signal_ticker=None,        # e.g. 'BTC-USD'
                 signal_ema_period=None,    # EMA period on signal ticker (None = ema_trend)
                 # Intraday SL/TP simulation — resolves hit-order within daily bars
                 # '1h' recommended for GBTC/ETFs; None = daily OHLC only (default)
                 sl_tp_tf=None,
                 # Entry day filter: set of weekday ints (0=Mon … 6=Sun)
                 # None = trade every day; {5,6} = weekends only; {0,1,2,3,4} = weekdays only
                 # Exits always execute regardless of day.
                 entry_days=None,
                 # Handoff params: if set and a position was opened on a weekend day
                 # (dayofweek >= 4), on the first weekday bar recompute TP and trail_atr
                 # using these values. Allows weekend trades to continue with weekday-scale
                 # exits. Dict with keys: 'trail_atr', 'tp_mult', 'max_hold_trend',
                 # 'max_hold_mr', 'max_hold_short'. Missing keys fall back to instance vals.
                 handoff_params=None,
                 # Allow long entries in 'transition' regime (mixed signals).
                 # Capped at max_pos=1 and requires min_strength_bear conviction.
                 # Useful for catching early recoveries from bear markets.
                 allow_transition_longs=False):
        self.ticker = ticker
        self.start  = start
        self.end    = end
        # Numeric params — exposed for V6 grid-search
        self.trail_atr       = trail_atr
        self.vol_target      = vol_target
        self.tp_mult         = tp_mult
        self.partial_tp_mult = partial_tp_mult
        self.dd_reduce       = dd_reduce
        self.dd_halt         = dd_halt
        # Signal/indicator params — per-asset tunable (V7)
        self.rsi_period        = rsi_period
        self.rsi_oversold      = rsi_oversold
        self.rsi_overbought    = rsi_overbought
        self.atr_period        = atr_period
        self.ema_trend         = ema_trend
        self.adx_thresh        = adx_thresh
        self.min_strength_up   = min_strength_up
        self.min_strength_bear = min_strength_bear
        # V8 execution structure params
        self.trail_cushion    = trail_cushion
        self.post_partial_mult = post_partial_mult
        self.macd_fast        = macd_fast
        self.macd_slow        = macd_slow
        self.macd_sig         = macd_sig
        self.max_hold_trend   = max_hold_trend
        self.max_hold_mr      = max_hold_mr
        # V9 new params
        self.di_filter        = di_filter
        self.obv_filter       = obv_filter
        self.enable_stoch_rsi = enable_stoch_rsi
        self.enable_bb_signal = enable_bb_signal
        self.partial_qty_pct  = partial_qty_pct
        self.vol_regime_scale = vol_regime_scale
        self.min_vol_ratio    = min_vol_ratio
        self.reentry_cooldown = reentry_cooldown
        # V10: short-selling
        self.allow_shorts   = allow_shorts
        self.max_hold_short = max_hold_short if max_hold_short is not None else max_hold_trend
        # V10: on-chain
        self.use_onchain       = use_onchain
        self.mvrv_long_thresh  = mvrv_long_thresh
        self.mvrv_short_thresh = mvrv_short_thresh
        self.fg_fear_thresh    = fg_fear_thresh
        self.fg_greed_thresh   = fg_greed_thresh
        self.signal_ticker     = signal_ticker
        self.signal_ema_period = signal_ema_period
        self._signal_emas      = None  # populated by _fetch_signal_emas()
        self.sl_tp_tf          = sl_tp_tf
        self._intraday_bars: dict = {}   # date -> [(H, L), ...] for SL/TP sequencing
        self.entry_days            = set(entry_days) if entry_days is not None else None
        self.handoff_params        = handoff_params  # None or dict of exit params for weekday handoff
        self.allow_transition_longs = allow_transition_longs
        self.data         = None
        self.vix          = None
        self.equity       = 100_000
        self.positions    = []
        self.closed_trades= []
        self.equity_curve = [100_000]

    # ------------------------------------------------------------------
    # DATA
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten_multiindex(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Flatten a MultiIndex columns DataFrame returned by yfinance."""
        if not isinstance(df.columns, pd.MultiIndex):
            return df
        names = df.columns.names
        ticker_level = names.index('Ticker') if 'Ticker' in names else 1
        try:
            return df.xs(ticker, level=ticker_level, axis=1)
        except KeyError:
            df.columns = df.columns.get_level_values(0)
            return df

    def fetch_data(self):
        try:
            raw = yf.download(self.ticker, start=self.start, end=self.end,
                              progress=False, auto_adjust=True)
            self.data = self._flatten_multiindex(raw, self.ticker)

            vix_raw = yf.download('^VIX', start=self.start, end=self.end,
                                  progress=False, auto_adjust=True)
            self.vix = self._flatten_multiindex(vix_raw, '^VIX')

            logger.info(f"Loaded {len(self.data)} bars of {self.ticker}")

            # V10: 24/7 leading EMA signal source
            if self.signal_ticker:
                self._fetch_signal_emas()

            # V10: on-chain data fetch (MVRV + Fear & Greed)
            if self.use_onchain:
                self._fetch_onchain_data()

            # Intraday SL/TP simulation bars
            if self.sl_tp_tf:
                self._fetch_intraday_sl_tp()

            return True
        except Exception as e:
            logger.error(f"Data fetch error: {e}")
            return False

    def _fetch_signal_emas(self):
        """Fetch signal_ticker (e.g. BTC-USD) and pre-compute regime states.
        Stores BTC-USD LT_Trend (1/-1), EMA50, EMA200, and slope% — all within BTC\'s
        own price scale so crossover comparisons remain valid. These feed Lead_* columns
        directly; GBTC\'s native EMA columns are NOT overridden."""
        period = self.signal_ema_period if self.signal_ema_period else self.ema_trend
        try:
            raw = yf.download(self.signal_ticker, start=self.start, end=self.end,
                              progress=False, auto_adjust=True)
            src = self._flatten_multiindex(raw, self.signal_ticker)
            src = src[['Close']].copy()
            src['EMA200'] = src['Close'].ewm(span=200,    adjust=False).mean()
            src['EMA50']  = src['Close'].ewm(span=period, adjust=False).mean()
            # Regime computed within BTC-USD\'s own price scale (valid comparisons)
            src['LT_Trend']  = np.where(src['Close'] > src['EMA200'], 1.0, -1.0)
            src['Slope_Pct'] = (
                (src['EMA50'] - src['EMA50'].shift(3))
                / src['EMA50'].shift(3).clip(lower=1e-6) * 100
            ).fillna(0.0)
            # Align to trading-ticker calendar (forward-fill weekends/holidays)
            aligned = src[['LT_Trend', 'EMA50', 'EMA200', 'Slope_Pct']].reindex(
                self.data.index, method='ffill'
            ).fillna({'LT_Trend': 1.0, 'EMA50': 0.0, 'EMA200': 1.0, 'Slope_Pct': 0.0})
            self._signal_emas = aligned
            logger.info(f"Signal regime from {self.signal_ticker} ({len(src)} bars, "
                        f"ema_period={period}) -> aligned to {len(self.data)} {self.ticker} bars")
        except Exception as e:
            logger.warning(f"Signal EMA fetch failed for {self.signal_ticker}: {e}")
            self._signal_emas = None

    def _fetch_intraday_sl_tp(self):
        """Fetch intraday candles to resolve SL/TP hit order within a daily bar.
        yfinance provides 1h data for ~730 days; older dates fall back to daily."""
        try:
            raw = yf.download(self.ticker, start=self.start, end=self.end,
                              interval=self.sl_tp_tf, progress=False, auto_adjust=True)
            src = self._flatten_multiindex(raw, self.ticker)
            bars: dict = {}
            for ts, row in src.iterrows():
                d = ts.date()
                if d not in bars:
                    bars[d] = []
                bars[d].append((float(row['High']), float(row['Low'])))
            self._intraday_bars = bars
            logger.info(f"Intraday {self.sl_tp_tf} for SL/TP: {len(bars)} days loaded")
        except Exception as e:
            logger.warning(f"Intraday SL/TP fetch failed ({self.sl_tp_tf}): {e}")
            self._intraday_bars = {}

    def _get_hl_sequence(self, date, daily_high, daily_low):
        """Return an ordered (H, L) sequence for exit resolution.
        With intraday bars loaded: returns them chronologically so SL vs TP
        priority is determined bar-by-bar (resolves ambiguity on volatile days).
        Without intraday data: returns a single (daily_high, daily_low) tuple
        so the backtest loop runs identically to the pre-intraday behaviour."""
        if not self._intraday_bars:
            return ((daily_high, daily_low),)
        d = date.date() if hasattr(date, 'date') else date
        bars = self._intraday_bars.get(d)
        return bars if bars else ((daily_high, daily_low),)

    def _fetch_onchain_data(self):
        """Fetch MVRV (CoinMetrics, free, 2011+) and Fear & Greed (alternative.me, 2018+)."""
        import json as _json

        # ── MVRV (Market Value / Realized Value) ──────────────────────────────
        try:
            start_str = self.start[:10]
            end_str   = self.end[:10]
            url = (f"https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
                   f"?assets=btc&metrics=CapMVRVCur&frequency=1d&page_size=10000"
                   f"&start_time={start_str}&end_time={end_str}")
            with urllib.request.urlopen(url, timeout=15) as r:
                raw = _json.loads(r.read())
            mvrv_data = {
                row['time'][:10]: float(row['CapMVRVCur'])
                for row in raw['data']
                if row.get('CapMVRVCur') is not None
            }
            mvrv_series = pd.Series(mvrv_data)
            mvrv_series.index = pd.to_datetime(mvrv_series.index)
            self._mvrv = mvrv_series.reindex(self.data.index).ffill().fillna(1.5)
            logger.info(f"On-chain MVRV loaded: {len(mvrv_data)} days")
        except Exception as e:
            logger.warning(f"MVRV fetch failed, using neutral 1.5: {e}")
            self._mvrv = pd.Series(1.5, index=self.data.index)

        # ── Fear & Greed Index ────────────────────────────────────────────────
        try:
            url2 = "https://api.alternative.me/fng/?limit=5000&format=json&date_format=us"
            with urllib.request.urlopen(url2, timeout=10) as r:
                raw2 = _json.loads(r.read())
            fg_data = {}
            for item in raw2['data']:
                try:
                    # date_format=us gives MM-DD-YYYY
                    ts = item['timestamp']
                    dt = pd.to_datetime(ts, format='%m-%d-%Y')
                    fg_data[dt] = int(item['value'])
                except Exception:
                    pass
            fg_series = pd.Series(fg_data).sort_index()
            self._fg = fg_series.reindex(self.data.index).ffill().fillna(50)
            logger.info(f"Fear & Greed loaded: {len(fg_data)} days")
        except Exception as e:
            logger.warning(f"Fear & Greed fetch failed, using neutral 50: {e}")
            self._fg = pd.Series(50, index=self.data.index)

    # ------------------------------------------------------------------
    # INDICATORS
    # ------------------------------------------------------------------
    def prepare_indicators(self):
        df = self.data.copy()

        # Trend structure — native (may be overridden by signal_ticker below)
        df['EMA200'] = df['Close'].ewm(span=200,            adjust=False).mean()
        df['EMA50']  = df['Close'].ewm(span=self.ema_trend, adjust=False).mean()
        df['EMA20']  = df['Close'].ewm(span=20,             adjust=False).mean()
        df['EMA9']   = df['Close'].ewm(span=9,              adjust=False).mean()

        # Long-term trend: price vs EMA200
        df['LT_Trend'] = np.where(df['Close'] > df['EMA200'], 1, -1)

        # EMA50 slope (3-day change to smooth noise)
        df['EMA50_Slope'] = df['EMA50'] - df['EMA50'].shift(3)

        # RSI — period tunable per asset
        delta = df['Close'].diff()
        gain  = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        df['RSI'] = 100 - (100 / (1 + gain / (loss + 1e-10)))

        # MACD — periods tunable per asset (V8)
        ema12 = df['Close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema26 = df['Close'].ewm(span=self.macd_slow, adjust=False).mean()
        df['MACD']      = ema12 - ema26
        df['Signal']    = df['MACD'].ewm(span=self.macd_sig, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal']

        # ATR(14)
        tr = pd.concat([
            df['High'] - df['Low'],
            (df['High'] - df['Close'].shift()).abs(),
            (df['Low']  - df['Close'].shift()).abs(),
        ], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(self.atr_period).mean()  # period tunable per asset

        # ADX — same period as ATR (standard convention)
        high_diff = df['High'].diff()
        low_diff  = -df['Low'].diff()
        plus_dm   = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm  = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        atr_safe  = np.clip(df['ATR'], 0.0001, None)
        plus_di   = 100 * pd.Series(plus_dm,  index=df.index).rolling(self.atr_period).mean() / atr_safe
        minus_di  = 100 * pd.Series(minus_dm, index=df.index).rolling(self.atr_period).mean() / atr_safe
        denom     = np.clip(plus_di + minus_di, 0.0001, None)
        dx        = 100 * (plus_di - minus_di).abs() / denom
        df['ADX'] = dx.rolling(self.atr_period).mean()

        # V9: store DI+/DI- for directional gating
        df['Plus_DI']  = plus_di.fillna(0)
        df['Minus_DI'] = minus_di.fillna(0)

        # V9: OBV 5-day slope (positive = accumulation)
        _obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
        df['OBV_Slope'] = (_obv - _obv.shift(5)).fillna(0)

        # V9: Stochastic RSI_K (0=oversold, 1=overbought)
        # Lookback must match rsi_period — using a different period (e.g. 14) on a
        # 9-period RSI produces a non-standard StochRSI and misleading oversold thresholds.
        _rsi_min = df['RSI'].rolling(self.rsi_period).min()
        _rsi_max = df['RSI'].rolling(self.rsi_period).max()
        df['StochRSI_K'] = ((df['RSI'] - _rsi_min) / (_rsi_max - _rsi_min + 1e-10)).fillna(0.5)

        # V9: Bollinger Band %B (0=at lower band, 1=at upper band)
        _bb_std = df['Close'].rolling(20).std()
        _bb_mid = df['Close'].rolling(20).mean()
        df['BB_PCT'] = ((df['Close'] - (_bb_mid - 2*_bb_std)) / (4*_bb_std + 1e-10)).fillna(0.5)

        # Volume
        df['Vol_Avg']   = df['Volume'].rolling(20).mean()
        df['Vol_Ratio'] = df['Volume'] / (df['Vol_Avg'] + 1)

        # 20-day range position
        df['High_20']     = df['High'].rolling(20).max()
        df['Low_20']      = df['Low'].rolling(20).min()
        df['Close_Range'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'] + 1e-10)

        # VIX
        df['VIX'] = self.vix['Close'].reindex(df.index).ffill().fillna(20)

        # V10: on-chain signals
        if self.use_onchain and hasattr(self, '_mvrv'):
            df['MVRV'] = self._mvrv.reindex(df.index).ffill().fillna(1.5)
            df['FG']   = self._fg.reindex(df.index).ffill().fillna(50)
        else:
            df['MVRV'] = 1.5   # neutral — on-chain disabled
            df['FG']   = 50

        # Realized volatility (20-day, annualised) for vol targeting
        df['Daily_Ret']    = df['Close'].pct_change()
        df['Realized_Vol'] = df['Daily_Ret'].rolling(20).std() * np.sqrt(252)

        # EMA50 recently crossed below EMA200 (death-cross within 10 bars)
        ema50_below_200 = (df['EMA50'] < df['EMA200']).astype(int)
        df['Death_Cross'] = (
            (ema50_below_200 == 1) &
            (ema50_below_200.shift(10).fillna(0) == 0)
        ).astype(int)

        # Golden cross: EMA50 recently crossed above EMA200
        ema50_above_200 = (df['EMA50'] > df['EMA200']).astype(int)
        df['Golden_Cross'] = (
            (ema50_above_200 == 1) &
            (ema50_above_200.shift(10).fillna(0) == 0)
        ).astype(int)

        # ── Lead_* regime columns ─────────────────────────────────────────────
        # When signal_ticker is set: BTC-USD\'s own EMA50/EMA200/LT_Trend feed
        # Lead_* (all in BTC\'s price scale, so EMA50>EMA200 comparison is valid).
        # GBTC\'s native EMA columns stay intact for buy/sell signal generation.
        # When no signal_ticker: mirrors native GBTC regime.
        if self._signal_emas is not None:
            sig = self._signal_emas.reindex(df.index).ffill()
            df['Lead_LT']        = sig['LT_Trend'].fillna(1.0)
            df['Lead_EMA50']     = sig['EMA50'].fillna(0.0)
            df['Lead_EMA200']    = sig['EMA200'].fillna(1.0)
            df['Lead_Slope_Pct'] = sig['Slope_Pct'].fillna(0.0)
        else:
            df['Lead_LT']       = df['LT_Trend'].astype(float)
            df['Lead_EMA50']    = df['EMA50']
            df['Lead_EMA200']   = df['EMA200']
            df['Lead_Slope_Pct'] = (
                (df['EMA50'] - df['EMA50'].shift(3)) / df['EMA50'].shift(3).clip(lower=1e-6) * 100
            ).fillna(0.0)

        self.data = df

    # ------------------------------------------------------------------
    # SIGNALS — LONG
    # ------------------------------------------------------------------
    def generate_buy_signals(self, idx):
        if idx < 200:
            return [], 0.0, 'trend'

        row   = self.data.iloc[idx]
        prev  = self.data.iloc[idx - 1]
        prev2 = self.data.iloc[idx - 2]
        prev3 = self.data.iloc[idx - 3]

        signals  = []
        strength = 0.0
        sig_type = 'trend'   # 'trend' → 60d hold; 'mr' → 25d hold

        # ---- Trend-following signals (sig_type = 'trend') ----
        if row['MACD'] > row['Signal'] > 0:
            signals.append('macd_positive');     strength += 0.25

        if row['EMA9'] > row['EMA20']:
            signals.append('ema9_above_20');     strength += 0.15

        if row['Close'] > row['EMA20'] > row['EMA50'] > row['EMA200']:
            signals.append('all_emas_aligned');  strength += 0.30

        if row['ADX'] > self.adx_thresh:
            signals.append('strong_trend');      strength += 0.15

        # Golden-cross buy: EMA50 just crossed above EMA200 — regime shift
        if row['Golden_Cross'] == 1 and row['MACD'] > row['Signal']:
            signals.append('golden_cross');      strength += 0.40

        # ---- Mean-reversion signals → override sig_type ----
        mr_strength = 0.0

        if (prev['RSI'] < self.rsi_oversold
                and self.rsi_oversold < row['RSI'] < self.rsi_oversold + 10):
            signals.append('rsi_oversold');      mr_strength += 0.35

        if row['Close_Range'] < 0.2 and row['Close'] > prev['Close']:
            signals.append('price_low_bounce');  mr_strength += 0.25

        if prev['MACD_Hist'] < 0 < row['MACD_Hist']:
            signals.append('macd_hist_pos');     mr_strength += 0.20

        if row['Vol_Ratio'] > 1.3:
            signals.append('volume_surge');      mr_strength += 0.10

        if prev['Close'] < prev['EMA20'] and row['Close'] > row['EMA20']:
            signals.append('ma_breakout');       mr_strength += 0.25

        # EMA20 dip-hold: price pulled back to EMA20, bouncing with trend
        ema20_touch  = abs(prev['Close'] - prev['EMA20']) / max(prev['EMA20'], 1) < 0.012
        ema20_rising = row['EMA20'] > prev['EMA20']
        rsi_reset    = 40 <= row['RSI'] <= 58
        if ema20_touch and ema20_rising and rsi_reset and row['Close'] > row['EMA20']:
            signals.append('ema20_dip_hold');    mr_strength += 0.35

        # 3-day pullback recovery
        three_lower = (prev3['Close'] > prev2['Close'] > prev['Close'])
        if three_lower and row['Close'] > prev['Close'] and row['RSI'] < 58:
            signals.append('pullback_3day');     mr_strength += 0.30

        # V9: Stochastic RSI oversold — more sensitive than plain RSI
        if self.enable_stoch_rsi and row['StochRSI_K'] < 0.20 and row['Close'] > prev['Close']:
            signals.append('stoch_rsi_oversold'); mr_strength += 0.30

        # V9: Bollinger Band lower-band bounce — price near lower BB with uptick
        if self.enable_bb_signal and row['BB_PCT'] < 0.25 and row['Close'] > prev['Close']:
            signals.append('bb_lower_bounce');    mr_strength += 0.25

        # V10: On-chain signals (MVRV + Fear & Greed)
        if self.use_onchain:
            mvrv = row['MVRV']
            fg   = row['FG']
            # MVRV below long_thresh = price trading below realized cap = strong buy zone
            if mvrv < self.mvrv_long_thresh:
                signals.append('mvrv_undervalued');  mr_strength += 0.40
            # Extreme Fear = contrarian buy signal
            if fg < self.fg_fear_thresh:
                signals.append('fg_extreme_fear');   mr_strength += 0.25
            # MVRV very high = suppress long entries (block overbought accumulation)
            if mvrv > self.mvrv_short_thresh:
                # Reduce long strength when on-chain says overvalued
                strength   *= 0.60
                mr_strength *= 0.60

        # If MR strength exceeds trend strength, classify as MR
        if mr_strength > strength:
            strength = mr_strength
            sig_type = 'mr'
        else:
            strength += mr_strength  # combined (trend + MR confirmation)

        return signals, strength, sig_type

    # ------------------------------------------------------------------
    # SIGNALS — SHORT
    # ------------------------------------------------------------------
    def generate_sell_signals(self, idx):
        if idx < 200:
            return [], 0.0, 'trend'

        row  = self.data.iloc[idx]
        prev = self.data.iloc[idx - 1]

        signals  = []
        strength = 0.0
        sig_type = 'trend'

        # ---- Trend-following shorts ----
        if row['MACD'] < row['Signal'] < 0:
            signals.append('macd_negative');      strength += 0.25

        if row['EMA9'] < row['EMA20']:
            signals.append('ema9_below_20');      strength += 0.15

        if row['ADX'] > self.adx_thresh:
            signals.append('strong_trend');       strength += 0.15

        # Death-cross short: EMA50 just crossed below EMA200 — major regime shift
        if row['Death_Cross'] == 1 and row['MACD'] < row['Signal']:
            signals.append('death_cross_combo'); strength += 0.45

        # Bear RSI rollover: RSI bounced to 60+ (overbought in downtrend) then
        # rolls over — high-quality short entry into supply zone
        if (prev['RSI'] > self.rsi_overbought - 5
                and row['RSI'] < prev['RSI'] and row['RSI'] > 48):
            signals.append('bear_rsi_rollover'); strength += 0.35

        # ---- Mean-reversion shorts ----
        mr_strength = 0.0

        if (prev['RSI'] > self.rsi_overbought
                and row['RSI'] < self.rsi_overbought
                and row['RSI'] > self.rsi_overbought - 10):
            signals.append('rsi_overbought');     mr_strength += 0.35

        if row['Close_Range'] > 0.8 and row['Close'] < prev['Close']:
            signals.append('price_high_roll');    mr_strength += 0.25

        if prev['MACD_Hist'] > 0 > row['MACD_Hist']:
            signals.append('macd_hist_neg');      mr_strength += 0.20

        if row['Vol_Ratio'] > 1.3:
            signals.append('volume_surge');       mr_strength += 0.10

        if prev['Close'] > prev['EMA20'] and row['Close'] < row['EMA20']:
            signals.append('ma_breakdown');       mr_strength += 0.25

        # V10: on-chain short boosts
        if self.use_onchain:
            mvrv = row['MVRV']
            fg   = row['FG']
            # MVRV above short_thresh = on-chain says overvalued — boost shorts
            if mvrv > self.mvrv_short_thresh:
                signals.append('mvrv_overvalued'); strength += 0.35
            # Extreme Greed = contrarian short signal
            if fg > self.fg_greed_thresh:
                signals.append('fg_extreme_greed'); strength += 0.20
            # NOTE: intentionally no short suppression for low MVRV —
            # Bitcoin's biggest crashes happen as MVRV falls through the floor

        if mr_strength > strength:
            strength = mr_strength
            sig_type = 'mr'
        else:
            strength += mr_strength

        return signals, strength, sig_type

    # ------------------------------------------------------------------
    # BACKTEST — lookahead-bias free
    # Signals on close[t] → fill at open[t+1]
    # SL/TP checked against intraday Low / High
    # ------------------------------------------------------------------
    def backtest(self):
        logger.info("Starting V6 10yr-robust backtest (no lookahead)...")
        # Reset state so backtest() can be called repeatedly on the same instance
        self.equity        = 100_000
        self.positions     = []
        self.closed_trades = []
        self.equity_curve  = [100_000]
        self.prepare_indicators()

        VOL_TARGET    = self.vol_target
        DD_REDUCE     = self.dd_reduce
        DD_HALT       = self.dd_halt
        ALLOW_SHORTS  = self.allow_shorts  # V10: controllable via param

        # Pre-extract hot-path columns as numpy arrays (major speedup vs per-bar .iloc)
        _open  = self.data['Open'].to_numpy(dtype=float)
        _high  = self.data['High'].to_numpy(dtype=float)
        _low   = self.data['Low'].to_numpy(dtype=float)
        _close = self.data['Close'].to_numpy(dtype=float)
        _atr   = self.data['ATR'].to_numpy(dtype=float)
        _vix   = self.data['VIX'].to_numpy(dtype=float)
        _adx   = self.data['ADX'].to_numpy(dtype=float)
        _lt    = self.data['LT_Trend'].to_numpy(dtype=float)
        _ema50 = self.data['EMA50'].to_numpy(dtype=float)
        _ema200= self.data['EMA200'].to_numpy(dtype=float)
        _rvol  = self.data['Realized_Vol'].to_numpy(dtype=float)
        _pdi   = self.data['Plus_DI'].to_numpy(dtype=float)
        _mdi   = self.data['Minus_DI'].to_numpy(dtype=float)
        _obvs  = self.data['OBV_Slope'].to_numpy(dtype=float)
        _volr  = self.data['Vol_Ratio'].to_numpy(dtype=float)
        _dates = self.data.index
        # BTC-leading regime arrays (same as native when lead_ema_df is None)
        _lead_lt   = self.data['Lead_LT'].to_numpy(dtype=float)
        _lead_ema50= self.data['Lead_EMA50'].to_numpy(dtype=float)
        _lead_ema200=self.data['Lead_EMA200'].to_numpy(dtype=float)
        _lead_slp  = self.data['Lead_Slope_Pct'].to_numpy(dtype=float)

        pending_entries = []
        peak_equity     = self.equity
        last_stop_bar   = -999   # V9: reentry cooldown tracker

        for idx in range(200, len(self.data)):
            date         = _dates[idx]
            open_        = _open[idx]
            high         = _high[idx]
            low          = _low[idx]
            close        = _close[idx]
            atr          = _atr[idx]
            vix          = _vix[idx]
            adx          = _adx[idx]
            lt           = _lt[idx]
            ema50        = _ema50[idx]
            ema200       = _ema200[idx]
            realized_vol = _rvol[idx]
            # V9: directional + volume indicators for entry gates
            plus_di_val   = _pdi[idx]
            minus_di_val  = _mdi[idx]
            obv_slope_val = _obvs[idx]
            vol_ratio_val = _volr[idx]

            # ── Three-regime classification ────────────────────────────
            # Uses BTC-leading EMA if lead_ema_df is set, otherwise native EMAs.
            # lead_lt / lead_ema50 / lead_ema200 are scale-matched to each other
            # (both from BTC-USD), so ema50 > ema200 comparison is valid.
            lead_lt        = _lead_lt[idx]
            lead_ema50     = _lead_ema50[idx]
            lead_ema200    = _lead_ema200[idx]
            lead_slope_pct = _lead_slp[idx]   # % per 3 days; threshold: > -0.05%

            # Confirmed uptrend:   price > EMA200 AND EMA50 > EMA200
            # Confirmed downtrend: price < EMA200 AND EMA50 < EMA200
            # Transitional:        mixed signals (bear rallies, recoveries)
            if lead_lt >= 0 and lead_ema50 > lead_ema200:
                trend_regime = 'up'
            elif lead_lt < 0 and lead_ema50 < lead_ema200:
                trend_regime = 'down'
            else:
                trend_regime = 'transition'

            # ── Volatility scaling ─────────────────────────────────────
            # Strong uptrend: allow gentle upscaling (1.2×)
            # Other: cap at 1.0× (protect against over-sizing in volatile regime)
            # V9: regime-adaptive vol target (uptrend → scale up slightly)
            _vol_tgt = VOL_TARGET * (self.vol_regime_scale if trend_regime == 'up' else 1.0)
            max_vol_scale = 1.2 if (trend_regime == 'up' and adx > 28 and vix < 20) else 1.0
            vol_scale = float(np.clip(
                _vol_tgt / max(realized_vol, 0.05), 0.4, max_vol_scale
            ))

            # ── Drawdown state ─────────────────────────────────────────
            peak_equity = max(peak_equity, self.equity)
            current_dd  = (peak_equity - self.equity) / peak_equity

            # ── 1. Fill pending entries at today's OPEN ────────────────
            for order in pending_entries:
                if len(self.positions) >= order.get('max_pos_at_entry', 8):
                    break
                entry  = open_
                tAtr   = self.trail_atr
                risk   = order['risk_amount']
                if order['side'] == 1:
                    sl         = entry - tAtr * atr
                    partial_tp = entry + self.partial_tp_mult * atr
                    tp         = entry + self.tp_mult * atr
                    qty        = int(risk / max(entry - sl, 0.01))
                else:
                    sl         = entry + tAtr * atr
                    partial_tp = entry - self.partial_tp_mult * atr
                    tp         = entry - self.tp_mult * atr
                    qty        = int(risk / max(sl - entry, 0.01))
                if qty > 0:
                    self.positions.append({
                        'date':           date,
                        'entry':          entry,
                        'qty':            qty,
                        'sl':             sl,
                        'tp':             tp,
                        'partial_tp':     partial_tp,
                        'partial_taken':  False,
                        'trail_atr':      tAtr,
                        'trail_active':   False,   # V6: delayed trail
                        'side':           order['side'],
                        'reason':         order['reason'],
                        'sig_type':       order['sig_type'],
                        'entry_dayofweek': date.dayofweek,  # for handoff detection
                        'handoff_done':   False,
                    })
            pending_entries = []

            # ── 2. Manage open positions ───────────────────────────────
            for pos in list(self.positions):
                exit_price  = None
                exit_reason = ''
                _is_short   = pos['side'] == -1

                # ── Weekday handoff: recompute exits with GBTC-scale params ──
                # Fires once on the first weekday bar after a weekend-opened position.
                if (self.handoff_params
                        and not pos.get('handoff_done', True)
                        and date.dayofweek < 5              # it's now a weekday
                        and pos.get('entry_dayofweek', 0) >= 4):  # opened Fri or later
                    hp = self.handoff_params
                    new_trail = hp.get('trail_atr', self.trail_atr)
                    new_tp    = hp.get('tp_mult',   self.tp_mult)
                    if pos['side'] == 1:
                        pos['tp'] = pos['entry'] + new_tp * atr
                    else:
                        pos['tp'] = pos['entry'] - new_tp * atr
                    pos['trail_atr'] = new_trail
                    # Override max_hold for the rest of this position's life
                    if _is_short:
                        pos['max_hold_override'] = hp.get('max_hold_short', self.max_hold_short)
                    elif pos.get('sig_type', 'trend') == 'trend':
                        pos['max_hold_override'] = hp.get('max_hold_trend', self.max_hold_trend)
                    else:
                        pos['max_hold_override'] = hp.get('max_hold_mr', self.max_hold_mr)
                    pos['handoff_done'] = True

                max_hold    = pos.get('max_hold_override',
                              self.max_hold_short if _is_short
                              else (self.max_hold_trend
                                    if pos.get('sig_type', 'trend') == 'trend'
                                    else self.max_hold_mr))

                if pos['side'] == 1:  # Long
                    for bar_h, bar_l in self._get_hl_sequence(date, high, low):
                        # Delayed trail activation
                        if not pos['trail_active']:
                            if bar_h >= pos['entry'] + self.trail_cushion * atr:
                                pos['trail_active'] = True
                        if pos['trail_active']:
                            new_trail = bar_h - pos['trail_atr'] * atr
                            if new_trail > pos['sl']:
                                pos['sl'] = new_trail
                        # SL hit
                        if bar_l <= pos['sl']:
                            exit_price  = pos['sl']
                            exit_reason = 'TrailSL'
                            break
                        # Full TP
                        elif bar_h >= pos['tp']:
                            exit_price  = pos['tp']
                            exit_reason = 'TP'
                            break
                        # Partial TP — no break, continue remaining bars
                        elif not pos['partial_taken'] and bar_h >= pos['partial_tp']:
                            partial_qty = max(1, int(pos['qty'] * self.partial_qty_pct))
                            if partial_qty > 0:
                                p_price     = pos['partial_tp']
                                partial_pnl = (p_price - pos['entry']) * partial_qty
                                self.equity += partial_pnl
                                self.closed_trades.append({
                                    'entry_date':  pos['date'], 'exit_date': date,
                                    'entry_price': pos['entry'], 'exit_price': p_price,
                                    'qty': partial_qty, 'side': 'LONG', 'pnl': partial_pnl,
                                    'pnl_pct': (partial_pnl/(pos['entry']*partial_qty))*100,
                                    'hold_days': (date-pos['date']).days,
                                    'reason': pos['reason'], 'exit_reason': 'PartialTP',
                                })
                                pos['qty']          -= partial_qty
                                pos['partial_taken'] = True
                                pos['tp']            = pos['entry'] + pos['trail_atr'] * atr * self.post_partial_mult
                                pos['sl']            = pos['entry']   # break-even
                    if exit_price is None and (date - pos['date']).days > max_hold:
                        exit_price  = close
                        exit_reason = 'Max_Hold'

                else:  # Short
                    for bar_h, bar_l in self._get_hl_sequence(date, high, low):
                        if not pos['trail_active']:
                            if bar_l <= pos['entry'] - self.trail_cushion * atr:
                                pos['trail_active'] = True
                        if pos['trail_active']:
                            new_trail = bar_l + pos['trail_atr'] * atr
                            if new_trail < pos['sl']:
                                pos['sl'] = new_trail
                        # SL hit
                        if bar_h >= pos['sl']:
                            exit_price  = pos['sl']
                            exit_reason = 'TrailSL'
                            break
                        # Full TP
                        elif bar_l <= pos['tp']:
                            exit_price  = pos['tp']
                            exit_reason = 'TP'
                            break
                        # Partial TP — no break, continue remaining bars
                        elif not pos['partial_taken'] and bar_l <= pos['partial_tp']:
                            partial_qty = max(1, int(pos['qty'] * self.partial_qty_pct))
                            if partial_qty > 0:
                                p_price     = pos['partial_tp']
                                partial_pnl = (pos['entry'] - p_price) * partial_qty
                                self.equity += partial_pnl
                                self.closed_trades.append({
                                    'entry_date':  pos['date'], 'exit_date': date,
                                    'entry_price': pos['entry'], 'exit_price': p_price,
                                    'qty': partial_qty, 'side': 'SHORT', 'pnl': partial_pnl,
                                    'pnl_pct': (partial_pnl/(pos['entry']*partial_qty))*100,
                                    'hold_days': (date-pos['date']).days,
                                    'reason': pos['reason'], 'exit_reason': 'PartialTP',
                                })
                                pos['qty']          -= partial_qty
                                pos['partial_taken'] = True
                                pos['tp']            = pos['entry'] - pos['trail_atr'] * atr * self.post_partial_mult
                                pos['sl']            = pos['entry']
                    if exit_price is None and (date - pos['date']).days > max_hold:
                        exit_price  = close
                        exit_reason = 'Max_Hold'

                if exit_price is not None:
                    pnl = ((exit_price - pos['entry']) * pos['qty'] if pos['side'] == 1
                           else (pos['entry'] - exit_price) * pos['qty'])
                    self.equity += pnl
                    self.closed_trades.append({
                        'entry_date':  pos['date'], 'exit_date': date,
                        'entry_price': pos['entry'], 'exit_price': exit_price,
                        'qty': pos['qty'],
                        'side': 'LONG' if pos['side'] == 1 else 'SHORT',
                        'pnl': pnl,
                        'pnl_pct': (pnl/(pos['entry']*pos['qty']+1e-10))*100,
                        'hold_days': (date-pos['date']).days,
                        'reason': pos['reason'], 'exit_reason': exit_reason,
                    })
                    self.positions.remove(pos)
                    # V9: track stop-outs for reentry cooldown
                    if exit_reason == 'TrailSL' and self.reentry_cooldown > 0:
                        last_stop_bar = idx

            self.equity_curve.append(self.equity)

            # ── Entry day filter ────────────────────────────────────────
            # Exits always run; only skip signal generation on non-allowed days.
            # date.dayofweek: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
            if self.entry_days is not None and date.dayofweek not in self.entry_days:
                continue

            # ── 3. Generate signals on close → queue for tomorrow ──────

            # ── Regime-adaptive position cap ──────────────────────────
            if trend_regime == 'up':
                if adx > self.adx_thresh + 3 and vix < 20:   max_pos = 8
                elif adx > self.adx_thresh - 5 or vix < 25:  max_pos = 4
                else:                                          max_pos = 2
            elif trend_regime == 'down':
                max_pos = 4 if adx > self.adx_thresh else 2
            else:
                max_pos = 1   # transitional: only highest conviction

            # ── VIX blocks — differentiated by side ───────────────────
            # Longs blocked above VIX 45 (extreme fear, no point buying dips)
            # Shorts blocked above VIX 60 (panic = capitulation, not the right short)
            long_vix_ok  = vix <= 45
            short_vix_ok = vix <= 60

            if current_dd >= DD_HALT or len(self.positions) >= max_pos:
                continue

            # ── Signal quality gate: adaptive by regime ────────────────
            #   Uptrend: lower bar to capture momentum  
            #   Down/transition: require conviction (only best setups)
            if trend_regime == 'up' and adx > self.adx_thresh + 3 and vix < 20:
                min_strength = max(0.15, self.min_strength_up - 0.05)
            elif trend_regime == 'up':
                min_strength = self.min_strength_up
            else:
                min_strength = self.min_strength_bear  # bear/transitional: stricter

            # ── LONG entries — CONFIRMED UPTREND ONLY ─────────────────
            # Require: Close > EMA200 AND EMA50 > EMA200 (trend_regime == 'up')
            # EMA50 must also be rising to prevent longs in dying bounces
            # V9: directional quality gates (only apply if enabled)
            _di_ok   = (not self.di_filter)  or (plus_di_val > minus_di_val)
            _obv_ok  = (not self.obv_filter) or (obv_slope_val > 0)
            _vol_ok  = (vol_ratio_val >= self.min_vol_ratio)
            _cool_ok = (idx - last_stop_bar) >= self.reentry_cooldown

            if trend_regime == 'up' and long_vix_ok and lead_slope_pct > -0.05 and _di_ok and _obv_ok and _vol_ok and _cool_ok:
                buy_sigs, buy_str, buy_type = self.generate_buy_signals(idx)
                if len(buy_sigs) >= 1 and buy_str >= min_strength:
                    if adx > self.adx_thresh + 3 and vix < 20:
                        base_risk = self.equity * 0.011
                    else:
                        base_risk = self.equity * 0.010
                    if vix < 15:    base_risk *= 1.10
                    elif vix > 25:  base_risk *= 0.80
                    risk_amount = base_risk * vol_scale
                    if current_dd >= DD_REDUCE:
                        risk_amount *= 0.50
                    pending_entries.append({
                        'side':              1,
                        'risk_amount':       risk_amount,
                        'reason':            ' + '.join(buy_sigs[:2]),
                        'sig_type':          buy_type,
                        'max_pos_at_entry':  max_pos,
                    })

            # ── TRANSITION LONG entries (optional) ────────────────────
            # Catch early recoveries: 1 position max, high-conviction MR only
            elif (self.allow_transition_longs
                  and trend_regime == 'transition'
                  and long_vix_ok
                  and lead_slope_pct > -0.05
                  and _di_ok and _obv_ok and _vol_ok and _cool_ok
                  and len(self.positions) == 0):  # only when flat
                buy_sigs, buy_str, buy_type = self.generate_buy_signals(idx)
                if len(buy_sigs) >= 1 and buy_str >= self.min_strength_bear:
                    risk_amount = self.equity * 0.008 * vol_scale  # reduced size in transition
                    if current_dd >= DD_REDUCE:
                        risk_amount *= 0.50
                    pending_entries.append({
                        'side':              1,
                        'risk_amount':       risk_amount,
                        'reason':            'TRANSITION:' + ' + '.join(buy_sigs[:2]),
                        'sig_type':          buy_type,
                        'max_pos_at_entry':  1,  # cap at 1 in transition
                    })

            # ── SHORT entries — CONFIRMED DOWNTREND ONLY ──────────────
            # Require: Close < EMA200 AND EMA50 < EMA200 (trend_regime == 'down')
            # Also require ADX > 18 to filter choppy sideways periods
            if ALLOW_SHORTS and trend_regime == 'down' and short_vix_ok and adx > max(self.adx_thresh - 7, 15):
                sell_sigs, sell_str, sell_type = self.generate_sell_signals(idx)
                short_min_strength = 0.30 if adx > self.adx_thresh else 0.40
                if len(sell_sigs) >= 1 and sell_str >= short_min_strength:
                    if adx > self.adx_thresh + 3:
                        base_risk = self.equity * 0.011
                    else:
                        base_risk = self.equity * 0.010
                    if vix < 15:    base_risk *= 1.10
                    elif vix > 25:  base_risk *= 0.80
                    risk_amount = base_risk * vol_scale
                    if current_dd >= DD_REDUCE:
                        risk_amount *= 0.50
                    pending_entries.append({
                        'side':              -1,
                        'risk_amount':       risk_amount,
                        'reason':            ' + '.join(sell_sigs[:2]),
                        'sig_type':          sell_type,
                        'max_pos_at_entry':  max_pos,
                    })

        # Close remaining positions at last bar
        last_close = float(self.data['Close'].iloc[-1])
        last_date  = self.data.index[-1]
        for pos in list(self.positions):
            pnl = ((last_close - pos['entry']) * pos['qty'] if pos['side'] == 1
                   else (pos['entry'] - last_close) * pos['qty'])
            self.equity += pnl
            self.closed_trades.append({
                'entry_date':  pos['date'], 'exit_date': last_date,
                'entry_price': pos['entry'], 'exit_price': last_close,
                'qty': pos['qty'],
                'side': 'LONG' if pos['side'] == 1 else 'SHORT',
                'pnl': pnl,
                'pnl_pct': (pnl/(pos['entry']*pos['qty']+1e-10))*100,
                'hold_days': (last_date - pos['date']).days,
                'reason': pos['reason'], 'exit_reason': 'EOB',
            })

        return self._metrics()

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------
    def _metrics(self):
        if not self.closed_trades:
            return {'error': 'No trades'}

        trades = pd.DataFrame(self.closed_trades)
        ret    = ((self.equity - 100_000) / 100_000) * 100
        wins   = int((trades['pnl'] > 0).sum())
        total  = len(trades)

        # Annualised Sharpe on daily equity returns
        equity_series = pd.Series(self.equity_curve)
        daily_ret     = equity_series.pct_change().dropna()
        sharpe = ((daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
                  if daily_ret.std() > 0 else 0.0)

        # Sortino ratio
        downside_sq = np.minimum(daily_ret.values, 0) ** 2
        semidev_ann = np.sqrt(downside_sq.mean()) * np.sqrt(252)
        mean_ann    = daily_ret.mean() * 252
        sortino     = (mean_ann / semidev_ann) if semidev_ann > 0 else 0.0

        # Per-trade Sharpe
        if total > 1:
            pnl_pcts     = trades['pnl_pct'].values
            trade_sharpe = float(np.mean(pnl_pcts) / (np.std(pnl_pcts) + 0.001))
        else:
            trade_sharpe = 0.0

        # Drawdown (vectorised — ~50× faster than Python loop)
        eq_arr = np.array(self.equity_curve, dtype=float)
        peak_arr = np.maximum.accumulate(eq_arr)
        max_dd = float(((peak_arr - eq_arr) / peak_arr * 100).max())

        # Profit factor
        gross_wins = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
        pf         = (gross_wins + 0.1) / (gross_loss + 0.1)

        # Calmar ratio
        years    = len(self.equity_curve) / 252
        cagr_val = ((self.equity / 100_000) ** (1 / max(years, 0.5)) - 1) * 100
        calmar   = cagr_val / max(max_dd, 0.01)

        # Long vs short breakdown
        longs  = trades[trades['side'] == 'LONG']
        shorts = trades[trades['side'] == 'SHORT']

        return {
            'symbol':        self.ticker,
            'return_pct':    round(ret, 2),
            'initial':       100_000,
            'final':         round(self.equity, 2),
            'years':         round(years, 1),
            'cagr':          round(cagr_val, 2),
            'trades':        total,
            'wins':          wins,
            'win_rate':      round((wins / total) * 100, 1) if total else 0,
            'sharpe':        round(sharpe, 2),
            'sortino':       round(sortino, 2),
            'calmar':        round(calmar, 2),
            'trade_sharpe':  round(trade_sharpe, 2),
            'avg_win_pct':   round(trades[trades['pnl'] > 0]['pnl_pct'].mean(), 2) if wins else 0,
            'avg_loss_pct':  round(trades[trades['pnl'] < 0]['pnl_pct'].mean(), 2) if (total-wins) else 0,
            'max_dd':        round(max_dd, 2),
            'profit_factor': round(pf, 2),
            'calmar_ratio':  round(calmar, 2),
            'best_trade':    round(float(trades['pnl'].max()), 2),
            'worst_trade':   round(float(trades['pnl'].min()), 2),
            'avg_hold_days': round(float(trades['hold_days'].mean()), 1),
            'long_trades':   len(longs),
            'short_trades':  len(shorts),
            'long_wr':       round((longs['pnl']>0).mean()*100, 1) if len(longs) else 0,
            'short_wr':      round((shorts['pnl']>0).mean()*100, 1) if len(shorts) else 0,
        }


# ---------------------------------------------------------------------------
# Entry point — quick SPY 10yr sanity check
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys

    ticker = sys.argv[1] if len(sys.argv) > 1 else 'SPY'
    logger.info("=" * 70)
    logger.info(f"AGGRESSIVE HYBRID TRADER V6 — 10yr ROBUST — {ticker}")
    logger.info("=" * 70)

    trader = AggressiveHybridV6(ticker)
    if trader.fetch_data():
        results = trader.backtest()

        print(f"\n{'=' * 70}")
        print(f"V6 10yr — {results['symbol']} ({results['years']:.1f} yrs)")
        print('=' * 70)
        print(f"Return:        {results['return_pct']:.2f}%")
        print(f"CAGR:          {results['cagr']:.2f}%")
        print(f"Trades:        {results['trades']}  ({results['trades']/results['years']:.0f}/yr)")
        print(f"  Longs:       {results['long_trades']}  WR {results['long_wr']:.1f}%")
        print(f"  Shorts:      {results['short_trades']}  WR {results['short_wr']:.1f}%")
        print(f"Sharpe:        {results['sharpe']:.2f}")
        print(f"Sortino:       {results['sortino']:.2f}")
        print(f"Calmar:        {results['calmar_ratio']:.2f}")
        print(f"Max DD:        {results['max_dd']:.2f}%")
        print(f"Win Rate:      {results['win_rate']:.1f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}x")
        print(f"Avg Hold:      {results['avg_hold_days']:.1f} days")
        print('=' * 70)

        tgt_sharpe = 1.5
        if results['sharpe'] >= tgt_sharpe:
            print(f"[OK] Sharpe {results['sharpe']:.2f} >= {tgt_sharpe} target over 10yr")
        else:
            print(f"[--] Sharpe {results['sharpe']:.2f} - room to improve")

        out = f"intraday/results/v6_{ticker.lower()}_10yr.json"
        with open(out, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Saved -> {out}")
