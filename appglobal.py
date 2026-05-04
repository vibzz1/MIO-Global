import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import concurrent.futures
import warnings
from functools import partial
from streamlit_lightweight_charts import renderLightweightCharts

warnings.filterwarnings("ignore")

# =============================================================================
# UI SETUP + CUSTOM STYLING
# =============================================================================
st.set_page_config(page_title="MIO Global Screener", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0e1117; }
    .grade-badge {
        display: inline-block; padding: 8px 18px; border-radius: 8px;
        font-size: 26px; font-weight: 800; text-align: center;
        color: white; margin-bottom: 8px; width: 100%;
    }
    .grade-aplus { background: linear-gradient(135deg, #00b060, #00d47e); }
    .grade-a { background: linear-gradient(135deg, #22c55e, #4ade80); color: #000; }
    .grade-b { background: linear-gradient(135deg, #eab308, #fbbf24); color: #000; }
    .grade-c { background: linear-gradient(135deg, #dc2626, #ef4444); }
    .score-bar-bg {
        background: #1e222d; border-radius: 6px; height: 18px;
        margin-bottom: 6px; overflow: hidden;
    }
    .score-bar-fill {
        height: 100%; border-radius: 6px; transition: width 0.5s ease;
        display: flex; align-items: center; justify-content: center;
        font-size: 11px; font-weight: 700; color: white;
    }
    .bar-green { background: linear-gradient(90deg, #00b060, #00d47e); }
    .bar-yellow { background: linear-gradient(90deg, #eab308, #fbbf24); }
    .bar-red { background: linear-gradient(90deg, #dc2626, #ef4444); }
    .detail-row {
        display: flex; justify-content: space-between; padding: 3px 0;
        font-size: 13px; border-bottom: 1px solid #1e222d;
    }
    .detail-label { color: #9ca3af; }
    .detail-value { color: #e5e7eb; font-weight: 600; }
    .section-divider {
        background: linear-gradient(90deg, #00b060, transparent);
        height: 3px; border-radius: 2px; margin: 20px 0 10px 0;
    }
    .trade-box {
        background: #1a1f2e; border: 1px solid #2d3748; border-radius: 10px;
        padding: 12px; margin-top: 8px;
    }
    .trade-entry { color: #4ade80; font-size: 18px; font-weight: 700; }
    .trade-sl { color: #ef4444; font-size: 18px; font-weight: 700; }
    .trade-risk { color: #fbbf24; font-size: 14px; }
    .market-flag { font-size: 28px; margin-right: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 🌍 MIO Global Setup Screener")
st.markdown("*Same scoring engine · 5 markets · Base · Stage · Timing*")

# =============================================================================
# MARKET CONFIGURATIONS
# =============================================================================
MARKETS = {
    "🇺🇸 US (S&P 500)": {
        "suffix": "",
        "currency": "$",
        "source": "sp500",
        "flag": "🇺🇸",
        "workers": 30,
    },
    "🇯🇵 Japan (Nikkei 225)": {
        "suffix": ".T",
        "currency": "¥",
        "source": "nikkei225",
        "flag": "🇯🇵",
        "workers": 20,
    },
    "🇰🇷 Korea (KOSPI 200)": {
        "suffix": ".KS",
        "currency": "₩",
        "source": "kospi200",
        "flag": "🇰🇷",
        "workers": 20,
    },
    "🇨🇳 China (CSI 300)": {
        "suffix": "",  # handled per-ticker (.SS or .SZ)
        "currency": "¥",
        "source": "csi300",
        "flag": "🇨🇳",
        "workers": 15,
    },
    "🇩🇪 Germany (HDAX 100)": {
        "suffix": ".DE",
        "currency": "€",
        "source": "germany",
        "flag": "🇩🇪",
        "workers": 20,
    },
}

# =============================================================================
# TICKER UNIVERSES — Wikipedia scrape + fallback curated lists
# =============================================================================

@st.cache_data(ttl=86400)
def get_sp500_tickers():
    """S&P 500 from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        sectors = dict(zip(tickers, df['GICS Sector']))
        return tickers, sectors
    except:
        # Fallback: top 100 liquid US stocks
        tickers = [
            "AAPL","MSFT","AMZN","NVDA","GOOGL","META","TSLA","BRK-B","UNH","JNJ",
            "XOM","JPM","V","PG","MA","HD","CVX","MRK","ABBV","LLY",
            "PEP","KO","COST","AVGO","TMO","MCD","WMT","CSCO","ACN","ABT",
            "DHR","NEE","LIN","PM","TXN","UNP","BMY","RTX","AMGN","HON",
            "LOW","QCOM","UPS","MS","GS","SCHW","BLK","ADP","INTC","AMD",
            "CAT","DE","BA","GE","MMM","IBM","DIS","CRM","ORCL","ADBE",
            "NOW","PYPL","SQ","SHOP","SNOW","PLTR","COIN","UBER","ABNB","DASH",
            "ZS","CRWD","PANW","DDOG","NET","MDB","TTWO","EA","ATVI","NFLX",
            "CMG","SBUX","NKE","LULU","TJX","ROST","DG","DLTR","WBA","CVS",
            "CI","HUM","ELV","MCK","CAH","ABC","SYK","BSX","MDT","ZBH",
        ]
        return tickers, {}


@st.cache_data(ttl=86400)
def get_nikkei225_tickers():
    """Nikkei 225 from Wikipedia."""
    try:
        url = "https://en.wikipedia.org/wiki/Nikkei_225"
        tables = pd.read_html(url)
        # Find the table with ticker codes
        for t in tables:
            cols = [str(c).lower() for c in t.columns]
            if any('code' in c or 'ticker' in c for c in cols):
                code_col = [c for c in t.columns if 'code' in str(c).lower() or 'ticker' in str(c).lower()][0]
                tickers = [str(int(x)) for x in t[code_col] if pd.notna(x)]
                sector_col = [c for c in t.columns if 'sector' in str(c).lower() or 'industry' in str(c).lower()]
                sectors = {}
                if sector_col:
                    sectors = dict(zip(tickers, t[sector_col[0]].astype(str)))
                return tickers, sectors
        raise ValueError("No ticker column found")
    except:
        # Fallback: major Nikkei components
        tickers = [
            "7203","6758","9984","8306","6861","6501","7267","4063","6902","8035",
            "9433","9432","7974","4502","4503","6098","7751","6367","8316","6954",
            "3382","7741","4568","6273","8411","9983","2914","4901","6326","8058",
            "8031","6702","7269","6503","4519","6752","5401","7011","6301","3407",
            "8601","2502","2802","4452","9020","9022","9021","1925","1928","5108",
            "6971","7272","8802","6506","4911","6645","7733","5713","3405","6988",
            "4543","8053","7731","9613","2801","4507","2503","8725","5802","6479",
            "4689","3086","8830","6762","5201","7912","4151","6113","7832","2413",
        ]
        return tickers, {}


@st.cache_data(ttl=86400)
def get_kospi200_tickers():
    """KOSPI 200 — curated list of major Korean stocks."""
    # KOSPI uses 6-digit codes on yfinance with .KS suffix
    tickers = [
        "005930","000660","005380","035420","051910","006400","068270","028260",
        "105560","055550","066570","003550","096770","012330","034730","015760",
        "032830","003670","009150","018260","033780","086790","036570","011170",
        "316140","352820","000270","005490","010950","017670","034020","030200",
        "035720","259960","247540","090430","323410","377300","003490","009540",
        "010120","010130","000810","004020","011780","001570","005830","021240","024110",
        "028050","032640","036460","042660","051900","053800","055660","064350",
        "069500","078930","086280","088350","097950","112040","139480","161390",
        "180640","192820","207940","214320","241560","251270","263750","271560",
        "282330","293490","302440","326030","336260","357780","373220","402340",
        "004170","006800","007070","010140","011070","011200","016360","018880",
        "020150","023530","029780","034220","035250","039490","044090","047050",
        "047810","052690","058470","060250","067630","069260","071050","078340",
    ]
    return tickers, {}


@st.cache_data(ttl=86400)
def get_csi300_tickers():
    """CSI 300 — major Chinese A-shares. Suffix baked in (.SS or .SZ)."""
    # Shanghai stocks: .SS, Shenzhen stocks: .SZ
    # These are already full yfinance symbols
    tickers = [
        "600519.SS","601318.SS","600036.SS","600276.SS","601166.SS","600030.SS",
        "601398.SS","600900.SS","600887.SS","601888.SS","600809.SS","601012.SS",
        "603259.SS","601628.SS","600585.SS","601668.SS","600031.SS","601601.SS",
        "600050.SS","601919.SS","600176.SS","600104.SS","601225.SS","600690.SS",
        "600048.SS","603501.SS","601688.SS","600436.SS","601816.SS","600183.SS",
        "601857.SS","600588.SS","601899.SS","600196.SS","603986.SS","601138.SS",
        "600346.SS","600703.SS","601390.SS","600029.SS","601088.SS","600989.SS",
        "601211.SS","600406.SS","600132.SS","601800.SS","603288.SS","600570.SS",
        "000858.SZ","000333.SZ","000651.SZ","000568.SZ","002714.SZ","300750.SZ",
        "002415.SZ","000725.SZ","002230.SZ","000001.SZ","002304.SZ","300059.SZ",
        "002352.SZ","000002.SZ","002475.SZ","300015.SZ","002049.SZ","000063.SZ",
        "002142.SZ","000776.SZ","002027.SZ","300124.SZ","002601.SZ","300274.SZ",
        "002594.SZ","300760.SZ","002032.SZ","300413.SZ","002460.SZ","300033.SZ",
        "002241.SZ","300122.SZ","000100.SZ","002311.SZ","000338.SZ","002466.SZ",
        "300347.SZ","002371.SZ","300529.SZ","000661.SZ","002050.SZ","300136.SZ",
    ]
    return tickers, {}


@st.cache_data(ttl=86400)
def get_germany_tickers():
    """HDAX 100 — top German stocks."""
    tickers = [
        "SAP","SIE","ALV","DTE","MBG","BMW","BAS","MUV2","DPW","IFX",
        "AIR","ADS","SHL","BEI","HEN3","VOW3","RWE","DB1","FRE","MRK",
        "HEI","EOAN","CON","1COV","SRT3","ZAL","PUM","LEG","QIA","FME",
        "MTX","HNR1","ENR","GXI","EVK","TKA","FPE3","RAA","BOSS","KGX",
        "DHL","HFG","DHER","SY1","KCO","LXS","HHFA","NDA","DEQ","AFX",
        "HAG","WAF","GFT","S92","AT1","AIXA","PNE","SBS","EVD","VNA",
        "RHM","DTG","FNTN","PSM","BNR","NDX1","G1A","HOT","TLX","KWS",
        "WCH","JUN3","NOEJ","SZU","TEG","SGL","FIE","COP","NEM","GLJ",
    ]
    return tickers, {}


def get_tickers_for_market(market_key):
    """Route to correct ticker fetch function."""
    cfg = MARKETS[market_key]
    source = cfg["source"]

    if source == "sp500":
        return get_sp500_tickers()
    elif source == "nikkei225":
        return get_nikkei225_tickers()
    elif source == "kospi200":
        return get_kospi200_tickers()
    elif source == "csi300":
        return get_csi300_tickers()
    elif source == "germany":
        return get_germany_tickers()
    return [], {}


def build_yf_symbol(ticker, market_key):
    """Build the full yfinance symbol."""
    cfg = MARKETS[market_key]
    source = cfg["source"]

    # China tickers already have suffix baked in
    if source == "csi300":
        return ticker

    suffix = cfg["suffix"]
    return f"{ticker}{suffix}" if suffix else ticker


# =============================================================================
# SCREENER LOGIC — IDENTICAL TO INDIA VERSION
# =============================================================================
def check_stock(ticker, market_key, ind_map):
    symbol = build_yf_symbol(ticker, market_key)
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(period="1y")
        if len(df) < 70:
            return None

        df.dropna(inplace=True)
        df['SMA_10'] = ta.sma(df['Close'], length=10)
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['ATR_20'] = ta.atr(df['High'], df['Low'], df['Close'], length=20)
        df['ATR_1'] = ta.true_range(df['High'], df['Low'], df['Close'])
        df['ADVOL_20'] = df['Volume'].rolling(20).mean()
        df['ADVOL_50'] = df['Volume'].rolling(50).mean()

        df.dropna(inplace=True)
        if len(df) < 22:
            return None

        sma50_trend_dn_20 = df['SMA_50'].iloc[-1] < df['SMA_50'].iloc[-21]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Volume thresholds adjust by market
        # US/Europe: 50K shares, Japan/Korea/China: 100K (different share structures)
        source = MARKETS[market_key]["source"]
        vol_threshold = 100000 if source in ["nikkei225", "kospi200", "csi300"] else 50000

        c1 = latest['ADVOL_20'] > vol_threshold
        c2 = latest['ADVOL_50'] > vol_threshold
        c3 = (df['SMA_20'].iloc[-5:] >= df['SMA_50'].iloc[-5:]).all()
        c4 = not (latest['Close'] < latest['SMA_50'] and sma50_trend_dn_20)
        c5 = latest['Close'] > latest['SMA_10']
        c6 = latest['Close'] > latest['SMA_20']
        c7 = latest['SMA_10'] > latest['SMA_20']
        c8 = latest['Close'] > prev['Close']
        c9 = latest['ATR_1'] > (latest['ATR_20'] * 0.6)
        c10 = latest['Close'] > (latest['Low'] + ((latest['High'] - latest['Low']) * 0.4))

        if all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]):
            industry = ind_map.get(ticker, None)
            if not industry or pd.isna(industry):
                try:
                    industry = stock.info.get('industry', 'N/A')
                except:
                    industry = 'N/A'
            return {"Ticker": ticker, "Symbol": symbol, "Industry": industry,
                    "chart_data": df, "market": market_key}

    except:
        pass
    return None


# =============================================================================
# SCORING ENGINE v3 — IDENTICAL TO INDIA (copy-pasted, battle-tested)
# =============================================================================

def _enrich_df(df):
    if 'SMA_150' not in df.columns:
        df['SMA_150'] = ta.sma(df['Close'], length=150)
    if 'SMA_200' not in df.columns:
        df['SMA_200'] = ta.sma(df['Close'], length=200)
    if 'ATR_5' not in df.columns:
        df['ATR_5'] = ta.atr(df['High'], df['Low'], df['Close'], length=5)
    if 'ATR_50' not in df.columns:
        df['ATR_50'] = ta.atr(df['High'], df['Low'], df['Close'], length=50)
    return df


def score_base_quality(df, latest):
    pre_trigger = df.iloc[-2]
    high_50 = df['High'].iloc[-51:-1].max()
    dist_from_peak = (high_50 - pre_trigger['Close']) / high_50 * 100

    if len(df) >= 7:
        atr_5_pre = ta.atr(df['High'].iloc[-7:-1], df['Low'].iloc[-7:-1],
                           df['Close'].iloc[-7:-1], length=5)
        atr_5_val = atr_5_pre.iloc[-1] if atr_5_pre is not None and len(atr_5_pre) > 0 else 0
    else:
        atr_5_val = 0
    atr_50_val = pre_trigger.get('ATR_50', atr_5_val)
    if pd.isna(atr_5_val): atr_5_val = 0
    if pd.isna(atr_50_val): atr_50_val = atr_5_val
    atr_contraction = (atr_50_val - atr_5_val) / atr_50_val * 100 if atr_50_val > 0 else 0

    vol_base = df['Volume'].iloc[-16:-1].mean()
    vol_prior = df['Volume'].iloc[-50:-15].mean() if len(df) >= 50 else vol_base
    vol_contraction = (vol_prior - vol_base) / vol_prior * 100 if vol_prior > 0 else 0

    high_100 = df['High'].iloc[-100:].max() if len(df) >= 100 else df['High'].max()
    pct_from_resistance = (high_100 - pre_trigger['Close']) / high_100 * 100

    low_200 = df['Low'].iloc[-min(len(df), 200):].min()
    prior_rally = (high_100 / low_200 - 1) * 100 if low_200 > 0 else 0
    depth_ratio = dist_from_peak / prior_rally if prior_rally > 0 else 1

    tight_days = sum(
        abs(df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1] < 0.015
        for i in range(-11, -1)
    )

    score = 0

    if depth_ratio <= 0.3 and dist_from_peak >= 3:
        score += 8
    elif 3 <= dist_from_peak <= 15:
        score += 8
    elif 1 <= dist_from_peak < 3:
        score += 6
    elif 15 < dist_from_peak <= 25:
        score += 5 if depth_ratio <= 0.4 else 4
    elif dist_from_peak < 1:
        score += 3
    elif 25 < dist_from_peak <= 50 and depth_ratio <= 0.3:
        score += 5
    else:
        score += 1

    if atr_contraction >= 30: score += 8
    elif atr_contraction >= 15: score += 6
    elif atr_contraction >= 0: score += 3
    else: score += 1

    if vol_contraction >= 40: score += 8
    elif vol_contraction >= 20: score += 6
    elif vol_contraction >= 5: score += 4
    else: score += 1

    if pct_from_resistance <= 5: score += 5
    elif pct_from_resistance <= 15: score += 3
    elif pct_from_resistance <= 30 and depth_ratio <= 0.3: score += 3
    else: score += 1

    score += min(tight_days, 4)
    score = min(score, 33)

    return score, {
        'Base Depth': f"{dist_from_peak:.1f}%",
        'ATR Contr': f"{atr_contraction:.0f}%",
        'Vol Contr': f"{vol_contraction:.0f}%",
        'Near Res': f"{pct_from_resistance:.0f}%",
        'Depth/Rally': f"{depth_ratio:.2f}",
        'Tight Days': tight_days
    }


def score_stage(df, latest):
    sma150 = latest.get('SMA_150', np.nan)
    sma200 = latest.get('SMA_200', np.nan)
    sma50 = latest['SMA_50']
    sma20 = latest['SMA_20']

    sma_stack_perfect = False
    sma_stack_good = latest['Close'] > sma20 > sma50

    if not pd.isna(sma150) and not pd.isna(sma200):
        sma_stack_perfect = (
            latest['Close'] > latest['SMA_10'] > sma20 > sma50 > sma150 > sma200
        )

    sma200_up = False
    sma200_flat = False
    sma200_dn = False
    sma200_val = np.nan
    has_200dma = False
    pct_change_200 = 0

    sma200_col = df['SMA_200'].dropna() if 'SMA_200' in df.columns else pd.Series(dtype=float)
    if len(sma200_col) >= 2:
        has_200dma = True
        sma200_val = sma200_col.iloc[-1]
        lookback = min(len(sma200_col) - 1, 21)
        pct_change_200 = (sma200_col.iloc[-1] / sma200_col.iloc[-1 - lookback] - 1) * 100
        sma200_up = pct_change_200 > 0.5
        sma200_flat = -0.5 <= pct_change_200 <= 0.5
        sma200_dn = pct_change_200 < -0.5

    if not has_200dma and 'SMA_150' in df.columns:
        sma150_col = df['SMA_150'].dropna()
        if len(sma150_col) >= 10:
            lookback = min(len(sma150_col) - 1, 21)
            pct_change_150 = (sma150_col.iloc[-1] / sma150_col.iloc[-1 - lookback] - 1) * 100
            sma200_up = pct_change_150 > 0.5
            sma200_flat = -0.5 <= pct_change_150 <= 0.5
            sma200_dn = pct_change_150 < -0.5
            sma200_val = sma150_col.iloc[-1]
            has_200dma = True
            pct_change_200 = pct_change_150

    price_above_200 = latest['Close'] > sma200_val if has_200dma else False
    sma50_above_200 = sma50 > sma200_val if has_200dma else False
    dist_from_200 = ((latest['Close'] - sma200_val) / sma200_val * 100) if has_200dma and sma200_val > 0 else 0

    sma200_flattening = has_200dma and sma200_dn and pct_change_200 > -2.0

    base_count = 0
    lookback_bars = min(len(df), 120)
    local_high = df['Close'].iloc[-lookback_bars]
    in_pullback = False

    for i in range(-lookback_bars, 0):
        close_i = df['Close'].iloc[i]
        if close_i > local_high:
            local_high = close_i
        pullback_pct = (local_high - close_i) / local_high * 100
        if pullback_pct >= 5 and not in_pullback:
            in_pullback = True
        elif in_pullback and close_i >= local_high * 0.97:
            base_count += 1
            in_pullback = False
            local_high = close_i

    low_200 = df['Low'].iloc[-min(len(df), 200):].min()
    high_recent = df['High'].iloc[-50:].max()
    prior_move_pct = (high_recent / low_200 - 1) * 100 if low_200 > 0 else 0

    move_period = df.iloc[-min(len(df), 120):-15]
    if len(move_period) > 10:
        up_days = (move_period['Close'] > move_period['Open']).sum()
        move_cleanliness = up_days / len(move_period) * 100
    else:
        move_cleanliness = 50

    is_early_s2 = sma200_up and price_above_200 and sma50_above_200 and base_count <= 1
    is_mid_s2 = sma200_up and price_above_200 and sma50_above_200 and base_count == 2
    is_late_s2 = sma200_up and price_above_200 and base_count >= 3
    is_s1b = (sma200_flat or sma200_flattening or (sma200_dn and price_above_200 and sma50_above_200)) and price_above_200
    is_s1_early = not sma200_up and not sma200_flat and price_above_200 and not is_s1b
    is_s4_s1 = not sma200_up and not price_above_200

    score = 0

    if sma_stack_perfect: score += 6
    elif sma_stack_good: score += 4
    else: score += 1

    if sma200_up: score += 6
    elif sma200_flat: score += 4
    elif sma200_flattening: score += 3
    else: score += 0

    if is_early_s2: score += 8
    elif is_s1b: score += 6
    elif is_mid_s2: score += 5
    elif is_late_s2: score += 2
    elif is_s1_early: score += 1
    else: score += 0

    if prior_move_pct >= 80: score += 8
    elif prior_move_pct >= 50: score += 6
    elif prior_move_pct >= 30: score += 4
    elif prior_move_pct >= 15: score += 2
    else: score += 0

    if move_cleanliness >= 55: score += 5
    elif move_cleanliness >= 50: score += 3
    else: score += 1

    if not sma200_up and not sma200_flat:
        if is_s1b and prior_move_pct >= 50:
            score = min(score, 22)
        elif sma200_flattening and prior_move_pct >= 30:
            score = min(score, 18)
        else:
            score = min(score, 12)

    if not price_above_200 and has_200dma: score -= 4
    if not sma50_above_200 and has_200dma: score -= 3
    if dist_from_200 > 35 and sma200_up: score -= 2

    score = max(min(score, 33), 0)

    if is_early_s2: stg_label = "S2·1st"
    elif is_s1b: stg_label = "S1b"
    elif is_mid_s2: stg_label = "S2·2nd"
    elif is_late_s2: stg_label = f"S2·{base_count}th"
    elif is_s1_early: stg_label = "S1"
    elif is_s4_s1: stg_label = "S4/1"
    else: stg_label = "?"

    stack_label = "Perfect" if sma_stack_perfect else ("Good" if sma_stack_good else "Weak")

    return score, {
        'MA Stack': stack_label,
        '200DMA': f"{'↑' if sma200_up else ('→' if sma200_flat else ('↗' if sma200_flattening else '↓'))} ({pct_change_200:+.1f}%)",
        'Stg': stg_label,
        'Bases': base_count,
        'Prior Move': f"{prior_move_pct:.0f}%",
        'Trend Clean': f"{move_cleanliness:.0f}%"
    }


def score_timing(df, latest):
    pct_20dma = ((latest['Close'] - latest['SMA_20']) / latest['SMA_20']) * 100
    abs_dist = abs(pct_20dma)

    if abs_dist <= 2: ma_pts = 10
    elif abs_dist <= 4: ma_pts = 7
    elif abs_dist <= 6: ma_pts = 4
    else: ma_pts = 1

    vol_today = latest['Volume']
    vol_base_avg = df['Volume'].iloc[-15:-1].mean()
    vol_expansion = (vol_today / vol_base_avg) if vol_base_avg > 0 else 1

    if vol_expansion >= 2.0: vol_pts = 8
    elif vol_expansion >= 1.5: vol_pts = 6
    elif vol_expansion >= 1.0: vol_pts = 3
    else: vol_pts = 1

    cr = latest['High'] - latest['Low']
    body = abs(latest['Close'] - latest['Open'])
    br = body / cr if cr > 0 else 0
    close_position = (latest['Close'] - latest['Low']) / cr if cr > 0 else 0.5

    if br > 0.6 and latest['Close'] > latest['Open'] and close_position > 0.7: candle_pts = 8
    elif br > 0.4 and latest['Close'] > latest['Open']: candle_pts = 5
    elif latest['Close'] > latest['Open']: candle_pts = 3
    else: candle_pts = 0

    high_50 = df['High'].iloc[-50:].max()
    pct_from_breakout = (high_50 - latest['Close']) / high_50 * 100

    days_above_resistance = 0
    for i in range(-10, 0):
        prior_high = df['High'].iloc[:len(df)+i-1].iloc[-50:].max()
        if df['Close'].iloc[i] > prior_high * 0.98:
            days_above_resistance += 1

    is_fresh_breakout = days_above_resistance <= 3 and latest['Close'] >= high_50 * 0.98
    is_stale_breakout = days_above_resistance > 5

    if is_fresh_breakout: bkout_pts = 8
    elif pct_from_breakout <= 3 and not is_stale_breakout: bkout_pts = 6
    elif pct_from_breakout <= 8: bkout_pts = 4
    elif is_stale_breakout: bkout_pts = 0
    else: bkout_pts = 1

    extension_penalty = 0
    is_volume_breakout = is_fresh_breakout and vol_expansion >= 1.5

    if not is_volume_breakout:
        if pct_20dma > 15: extension_penalty = 12
        elif pct_20dma > 10: extension_penalty = 8
        elif pct_20dma > 8: extension_penalty = 4
    else:
        if pct_20dma > 25: extension_penalty = 6
        elif pct_20dma > 20: extension_penalty = 3

    breakout_bonus = 0
    if is_volume_breakout and pct_20dma > 5:
        breakout_bonus = min(int(pct_20dma / 3), 6)

    raw_score = ma_pts + vol_pts + candle_pts + bkout_pts + breakout_bonus
    score = max(min(raw_score - extension_penalty, 34), 0)

    return score, {
        'Dist 20DMA': f"{pct_20dma:+.1f}%",
        'Vol Exp': f"{vol_expansion:.1f}x",
        'Body': f"{br:.0%}",
        'Close Pos': f"{close_position:.0%}",
        'Near Bkout': f"{pct_from_breakout:.1f}%",
        'Bkout Age': f"{days_above_resistance}d",
        'Fresh BO': "✅" if is_volume_breakout else "❌"
    }


def score_setup(res):
    df = _enrich_df(res['chart_data'].copy())
    latest = df.iloc[-1]

    base_s, base_d = score_base_quality(df, latest)
    stage_s, stage_d = score_stage(df, latest)
    timing_s, timing_d = score_timing(df, latest)
    total = base_s + stage_s + timing_s

    if total >= 75: grade = "A+"
    elif total >= 60: grade = "A"
    elif total >= 45: grade = "B"
    else: grade = "C"

    base_low = df['Low'].iloc[-50:].min()
    entry = latest['Close']
    sl = round(base_low * 0.995, 2)
    risk = round((entry - sl) / entry * 100, 1)

    return {'Grade': grade, 'Total': total, 'Base': base_s, 'Stage': stage_s,
            'Timing': timing_s, 'Entry': round(entry, 2), 'SL': sl, 'Risk%': risk,
            'base_det': base_d, 'stage_det': stage_d, 'timing_det': timing_d}


# =============================================================================
# TV-STYLE CHART RENDERER
# =============================================================================
def render_tv_chart(df, ticker, grade, total):
    candle_data = [{"time": idx.strftime("%Y-%m-%d"), "open": round(r['Open'], 2),
                    "high": round(r['High'], 2), "low": round(r['Low'], 2),
                    "close": round(r['Close'], 2)} for idx, r in df.iterrows()]

    volume_data = [{"time": idx.strftime("%Y-%m-%d"), "value": int(r['Volume']),
                    "color": "#26a69a80" if r['Close'] >= r['Open'] else "#ef535080"}
                   for idx, r in df.iterrows()]

    sma20_data = [{"time": idx.strftime("%Y-%m-%d"), "value": round(r['SMA_20'], 2)}
                  for idx, r in df.iterrows() if not pd.isna(r.get('SMA_20', np.nan))]

    chart_opts = [
        {
            "height": 400,
            "layout": {"background": {"type": "solid", "color": "#131722"},
                       "textColor": "#d1d4dc", "fontSize": 12},
            "grid": {"vertLines": {"color": "#1e222d"}, "horzLines": {"color": "#1e222d"}},
            "crosshair": {"mode": 0},
            "priceScale": {"borderColor": "#2d3748"},
            "timeScale": {"borderColor": "#2d3748", "timeVisible": False},
            "watermark": {"visible": True, "fontSize": 32, "horzAlign": "center",
                          "vertAlign": "center", "color": "rgba(255,255,255,0.03)",
                          "text": f"{ticker} | {grade} ({total})"}
        },
        {
            "height": 120,
            "layout": {"background": {"type": "solid", "color": "#131722"},
                       "textColor": "#d1d4dc", "fontSize": 10},
            "grid": {"vertLines": {"color": "#1e222d"}, "horzLines": {"color": "#1e222d"}},
            "timeScale": {"borderColor": "#2d3748", "timeVisible": False}
        }
    ]

    series_price = [
        {"type": "Candlestick", "data": candle_data,
         "options": {"upColor": "#26a69a", "downColor": "#ef5350",
                     "borderUpColor": "#26a69a", "borderDownColor": "#ef5350",
                     "wickUpColor": "#26a69a", "wickDownColor": "#ef5350"}},
        {"type": "Line", "data": sma20_data,
         "options": {"color": "#ff9800", "lineWidth": 2, "title": ""}}
    ]

    series_vol = [
        {"type": "Histogram", "data": volume_data,
         "options": {"priceFormat": {"type": "volume"}, "priceScaleId": "vol"}}
    ]

    renderLightweightCharts([
        {"chart": chart_opts[0], "series": series_price},
        {"chart": chart_opts[1], "series": series_vol}
    ], key=f"tv_{ticker}")


# =============================================================================
# SCORE PANEL RENDERER
# =============================================================================
def render_score_panel(res, currency="$"):
    grade = res.get('Grade', '?')
    total = res.get('Total', 0)
    grade_class = {'A+': 'grade-aplus', 'A': 'grade-a', 'B': 'grade-b', 'C': 'grade-c'}.get(grade, 'grade-c')

    st.markdown(f'<div class="grade-badge {grade_class}">{grade} · {total}/100</div>', unsafe_allow_html=True)

    dims = [('🧱 Base', res.get('Base', 0), 33),
            ('📶 Stage', res.get('Stage', 0), 33),
            ('⏱️ Timing', res.get('Timing', 0), 34)]

    for label, val, mx in dims:
        pct = int(val / mx * 100)
        bar_class = "bar-green" if pct >= 70 else "bar-yellow" if pct >= 50 else "bar-red"
        st.markdown(f"**{label}**")
        st.markdown(
            f'<div class="score-bar-bg">'
            f'<div class="score-bar-fill {bar_class}" style="width:{pct}%">{val}/{mx}</div>'
            f'</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top:8px">', unsafe_allow_html=True)
    all_details = {}
    for k in ['base_det', 'stage_det', 'timing_det']:
        all_details.update(res.get(k, {}))
    for label, value in all_details.items():
        st.markdown(f'<div class="detail-row"><span class="detail-label">{label}</span>'
                    f'<span class="detail-value">{value}</span></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="trade-box">'
        f'<div class="trade-entry">Entry {currency}{res.get("Entry", 0):,.2f}</div>'
        f'<div class="trade-sl">SL {currency}{res.get("SL", 0):,.2f}</div>'
        f'<div class="trade-risk">Risk {res.get("Risk%", 0)}%</div>'
        f'</div>', unsafe_allow_html=True)


# =============================================================================
# DASHBOARD
# =============================================================================
col_market, col_action = st.columns([3, 1])

with col_market:
    selected_markets = st.multiselect(
        "Select Markets:",
        list(MARKETS.keys()),
        default=["🇺🇸 US (S&P 500)"],
        help="Pick one or more markets to scan simultaneously"
    )

with col_action:
    st.markdown("<br>", unsafe_allow_html=True)
    run_scan = st.button("🚀 Run Global Scan", type="primary", use_container_width=True)

if run_scan and selected_markets:
    all_scored = []

    for market_key in selected_markets:
        cfg = MARKETS[market_key]
        flag = cfg["flag"]
        currency = cfg["currency"]

        st.markdown(f"### {flag} Scanning {market_key}...")
        tickers, ind_map = get_tickers_for_market(market_key)

        if not tickers:
            st.error(f"Failed to pull tickers for {market_key}")
            continue

        st.info(f"⚡ Scanning {len(tickers)} stocks...")

        passed_results = []
        progress_bar = st.progress(0)

        def _check(t, mk=market_key, im=ind_map):
            return check_stock(t, mk, im)

        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["workers"]) as executor:
            results = executor.map(_check, tickers)
            for i, result in enumerate(results):
                if result:
                    passed_results.append(result)
                progress_bar.progress(min((i + 1) / len(tickers), 1.0))
        progress_bar.empty()

        if passed_results:
            st.success(f"{flag} Found {len(passed_results)} setups in {market_key}")

            for res in passed_results:
                try:
                    scores = score_setup(res)
                    scored = {**res, **scores, 'currency': currency, 'flag': flag}
                    all_scored.append(scored)
                except:
                    all_scored.append({**res, 'Grade': '?', 'Total': 0, 'Base': 0,
                        'Stage': 0, 'Timing': 0, 'Entry': 0, 'SL': 0, 'Risk%': 0,
                        'base_det': {}, 'stage_det': {}, 'timing_det': {},
                        'currency': currency, 'flag': flag})
        else:
            st.warning(f"{flag} No setups found in {market_key}")

    # --- COMBINED RESULTS ---
    if all_scored:
        all_scored.sort(key=lambda x: x.get('Total', 0), reverse=True)

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader(f"📋 Global Ranked Scorecard ({len(all_scored)} setups)")

        table_data = []
        for s in all_scored:
            row = {
                '': s.get('flag', ''),
                'Ticker': s['Ticker'],
                'Industry': s['Industry'],
                'Grade': s.get('Grade', '?'),
                'Score': s.get('Total', 0),
                'Base': s.get('Base', 0),
                'Stage': s.get('Stage', 0),
                'Timing': s.get('Timing', 0),
                'CMP': s.get('Entry', 0),
                'SL': s.get('SL', 0),
                'Risk%': s.get('Risk%', 0),
            }
            for dk in ['base_det', 'stage_det', 'timing_det']:
                if s.get(dk):
                    row.update(s[dk])
            table_data.append(row)

        df_scored = pd.DataFrame(table_data)
        df_scored.index = df_scored.index + 1

        def color_grade(val):
            return {"A+": "background-color:#00b060;color:white;font-weight:bold",
                    "A": "background-color:#4ade80;color:black;font-weight:bold",
                    "B": "background-color:#fbbf24;color:black",
                    "C": "background-color:#ff333a;color:white"}.get(val, "")

        def color_score(val):
            try:
                v = int(val)
                if v >= 75: return "background-color:#00b060;color:white"
                elif v >= 60: return "background-color:#4ade80;color:black"
                elif v >= 45: return "background-color:#fbbf24;color:black"
                else: return "background-color:#ff333a;color:white"
            except: return ""

        styled = df_scored.style.map(color_grade, subset=['Grade']).map(color_score, subset=['Score'])
        st.dataframe(styled, use_container_width=True, height=500)

        # --- TV CHARTS ---
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.subheader("📈 Charts + Setup Scores (Global Rank)")

        for res in all_scored:
            grade = res.get('Grade', '?')
            total = res.get('Total', 0)
            currency = res.get('currency', '$')
            flag = res.get('flag', '')
            grade_emoji = {"A+": "🟢", "A": "🟡", "B": "🟠", "C": "🔴"}.get(grade, "⚪")

            col_chart, col_score = st.columns([3, 1])

            with col_chart:
                st.markdown(f"### {grade_emoji} {flag} **{res['Ticker']}** | {res['Industry']}")
                render_tv_chart(res['chart_data'], res['Ticker'], grade, total)

            with col_score:
                render_score_panel(res, currency)

            st.markdown("---")

elif run_scan:
    st.warning("Select at least one market to scan.")
