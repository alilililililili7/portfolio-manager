import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
import streamlit as st
import yfinance as yf

# ARCH/GARCH Modülü (Kurulu değilse fallback çalışır)
try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

st.set_page_config(
    page_title="Esnek & Dinamik Portföy Engine", layout="wide", page_icon="🏛️"
)

# ==========================================
# 1. YARDIMCI VE KANTİTATİF FONKSİYONLAR
# ==========================================


def fetch_data(tickers, period="3y"):
    """YFinance üzerinden kapanış verilerini çeker."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, period=period, progress=False)["Close"]
    if isinstance(data, pd.Series):
        data = data.to_frame()
    data = data.dropna(how="all").ffill().bfill()
    return data


def calculate_mpt_weights(returns_df, risk_free_rate=0.45):
    """Analitik Sharpe Optimizasyonu & Kovaryans Tersi (MPT Core)."""
    mean_returns = returns_df.mean() * 252
    cov_matrix = returns_df.cov() * 252

    # Kovaryans Tersi (Pinv)
    cov_inv = np.linalg.pinv(cov_matrix.values)
    ones = np.ones(len(mean_returns))

    # Excess Returns
    excess_returns = mean_returns.values - risk_free_rate

    # Analytical Unconstrained Weights
    raw_weights = np.dot(cov_inv, excess_returns)

    # Sıfırdan küçükleri engelle & Normalize et (Long-Only constraint)
    raw_weights = np.maximum(raw_weights, 0.025)  # Min %2.5 taban
    weights = raw_weights / np.sum(raw_weights)

    return pd.Series(weights, index=returns_df.columns), cov_matrix


def calculate_advanced_metrics(returns_df, weights, risk_free_rate=0.45):
    """VaR, CVaR, Skewness, Kurtosis hesaplamaları."""
    port_daily_returns = returns_df.dot(weights)

    # Risk & İstatistik
    skewness = stats.skew(port_daily_returns)
    kurtosis = stats.kurtosis(port_daily_returns)  # Excess Kurtosis

    # Parametrik & Historical VaR (%95 Güven)
    var_95 = np.percentile(port_daily_returns, 5)
    cvar_tail = port_daily_returns[port_daily_returns <= var_95]
    cvar_95 = (
        cvar_tail.mean() if len(cvar_tail) > 0 else var_95
    )  # Expected Shortfall

    # Yıllıklandırılmış Getiri & Volatilite
    ann_return = port_daily_returns.mean() * 252
    ann_vol = port_daily_returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol != 0 else 0

    return {
        "Ann_Return": ann_return,
        "Ann_Vol": ann_vol,
        "Sharpe": sharpe,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "VaR_95_Daily": abs(var_95),
        "CVaR_95_Daily": abs(cvar_95),
        "Daily_Returns": port_daily_returns,
    }


def calculate_ewma_cov(returns_df, lambda_param=0.94):
    """Dinamik EWMA Kovaryans Matrisi."""
    T, N = returns_df.shape
    returns = returns_df.values
    norm_returns = returns - np.mean(returns, axis=0)

    ewma_cov = np.zeros((N, N))
    for t in range(T):
        weight = (1 - lambda_param) * (lambda_param ** (T - 1 - t))
        ewma_cov += weight * np.outer(norm_returns[t], norm_returns[t])

    return pd.DataFrame(
        ewma_cov * 252, index=returns_df.columns, columns=returns_df.columns
    )


def run_garch_volatility(returns_series):
    """GARCH(1,1) Volatilite Tahmini."""
    if not ARCH_AVAILABLE:
        return returns_series.std() * np.sqrt(252)
    try:
        am = arch_model(
            returns_series * 100, vol="Garch", p=1, q=1, dist="Normal"
        )
        res = am.fit(disp="off")
        forecast_var = res.forecast(horizon=1).variance.iloc[-1, 0]
        annualized_garch_vol = (np.sqrt(forecast_var) / 100) * np.sqrt(252)
        return annualized_garch_vol
    except:
        return returns_series.std() * np.sqrt(252)


def run_monte_carlo_gbm(returns_df, weights, initial_budget, days=252, N=1000):
    """Cholesky Ayrıştırması ile Çok Değişkenli GBM Monte Carlo Simülasyonu."""
    port_returns = returns_df.dot(weights)
    mu = port_returns.mean()
    cov = returns_df.cov().values

    # Cholesky Decomposition
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Pozitif yarı-tanımlı değilse jitter ekle
        cov += np.eye(cov.shape[0]) * 1e-6
        L = np.linalg.cholesky(cov)

    num_assets = len(weights)
    w = weights.values

    # Simülasyon Patikaları
    simulation_paths = np.zeros((days, N))
    dt = 1

    for i in range(N):
        # Bağımsız rastgele normaller -> Korelasyonlu normallere dönüşüm
        Z = np.random.normal(size=(days, num_assets))
        correlated_Z = np.dot(Z, L.T)

        # Ağırlıklandırılmış Günlük Portföy Şoku
        port_shocks = np.dot(correlated_Z, w)
        daily_factors = 1 + (mu + port_shocks)

        # Fiyat Patikası
        price_path = initial_budget * np.cumprod(daily_factors)
        simulation_paths[:, i] = price_path

    return simulation_paths


def run_ornstein_uhlenbeck(
    initial_rate, target_rate=0.35, theta=0.1, sigma=0.05, days=252, N=200
):
    """Faiz Oranı Mean-Reversion (Vasicek / OU / CIR) Simülasyonu."""
    dt = 1 / days
    rates = np.zeros((days, N))
    rates[0, :] = initial_rate

    for t in range(1, days):
        dr = (
            theta * (target_rate - rates[t - 1, :]) * dt
            + sigma * np.sqrt(dt) * np.random.normal(size=N)
        )
        rates[t, :] = rates[t - 1, :] + dr

    return rates


# ==========================================
# 2. SIDEBAR (VARLIK VE BÜTÇE GİRDİLERİ)
# ==========================================
st.sidebar.title("💰 Bütçe ve Para Birimi")
currency = st.sidebar.selectbox(
    "Para Birimi Seçimi", ["TRY (₺) - TL Bazlı", "USD ($) - Dolar Bazlı"]
)
curr_symbol = "₺" if "TRY" in currency else "$"

budget = st.sidebar.number_input(
    f"Toplam Bütçe ({curr_symbol})", min_value=1000.0, value=100000.0, step=5000.0
)

st.sidebar.title("🎯 Varlık Seçimleri")

# Risk-Free Faiz
use_rf = st.sidebar.checkbox("🏛️ Risk-Free Faiz Oranını Kullan", value=True)
rf_rate_annual = (
    st.sidebar.number_input(
        "Yıllık Faiz / Mevduat Oranı (%)", value=45.0, step=1.0
    )
    / 100.0
    if use_rf
    else 0.0
)

# Hisse Senetleri
use_stocks = st.sidebar.checkbox("📈 Hisse Senetleri", value=True)
stock_tickers_input = st.sidebar.text_input(
    "Hisseler (Ticker)", "THYAO.IS, EREGL.IS, KCHOL.IS, TUPRS.IS, BIMAS.IS, AKBNK.IS, SISE.IS"
)

# ETF'ler
use_etfs = st.sidebar.checkbox("🧺 Tematik / Sektörel ETF'ler", value=False)
etf_tickers_input = st.sidebar.text_input("ETF'ler (Ticker)", "SPY, VOO")

# Emtia
use_commodities = st.sidebar.checkbox("🥇 Emtia & Değerli Madenler", value=False)
commodity_tickers_input = st.sidebar.text_input("Maden/Emtia (Ticker)", "IAU, CPER")

# GYO
use_reits = st.sidebar.checkbox("🏢 Emlak / GYO", value=False)
reit_tickers_input = st.sidebar.text_input("Emlak/GYO (Ticker)", "EKGYO.IS")

calculate_btn = st.sidebar.button("🚀 Portföyü Hesapla & Optimize Et")

# ==========================================
# 3. ANA SAYFA VE HESAPLAMA MOTORU
# ==========================================
st.title("🏛️ Esnek & Dinamik Portföy Optimizasyon Motoru")

# Seçili Ticker'ları Birleştir
selected_tickers = []
if use_stocks and stock_tickers_input:
    selected_tickers.extend([t.strip() for t in stock_tickers_input.split(",")])
if use_etfs and etf_tickers_input:
    selected_tickers.extend([t.strip() for t in etf_tickers_input.split(",")])
if use_commodities and commodity_tickers_input:
    selected_tickers.extend(
        [t.strip() for t in commodity_tickers_input.split(",")]
    )
if use_reits and reit_tickers_input:
    selected_tickers.extend([t.strip() for t in reit_tickers_input.split(",")])

if len(selected_tickers) < 2:
    st.warning(
        "Lütfen optimizasyon için en az 2 varlık seçin veya ticker girin."
    )
    st.stop()

# Veri Çekme ve Hesaplama
raw_data = fetch_data(selected_tickers)
returns_df = raw_data.pct_change().dropna()

# 1. MPT Ağırlık ve Kovaryans Tersi
optimal_weights, sample_cov = calculate_mpt_weights(
    returns_df, risk_free_rate=rf_rate_annual
)

# 2. Advanced Metrikler
metrics = calculate_advanced_metrics(
    returns_df, optimal_weights, risk_free_rate=rf_rate_annual
)

# ==========================================
# 1. GENİŞ PORTFÖY ÖZETLERİ
# ==========================================
st.subheader(f"📌 1. Geniş Portföy Özetleri ({curr_symbol} Bazlı)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Bütçe", f"{curr_symbol}{budget:,.2f}")
col2.metric("Beklenen Yıllık Getiri", f"%{metrics['Ann_Return']*100:.2f}")
col3.metric("Portföy Volatilite (Risk)", f"%{metrics['Ann_Vol']*100:.2f}")
col4.metric("Sharpe Oranı", f"{metrics['Sharpe']:.2f}")

st.divider()

# ==========================================
# 2. SEKTÖR / KATEGORİ VE VARLIK TABLOSU
# ==========================================
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.subheader("📋 2. Sektör / Kategori Bazlı Dağılım")
    # Basit Kategori Tanımlama
    categories = []
    for ticker in optimal_weights.index:
        if ".IS" in ticker:
            categories.append("Hisse (BİST)")
        elif ticker in ["SPY", "VOO"]:
            categories.append("ETF / ABD")
        elif ticker in ["IAU", "CPER"]:
            categories.append("Emtia")
        else:
            categories.append("Hisse / Diğer")

    alloc_df = pd.DataFrame(
        {
            "Ticker": optimal_weights.index,
            "Kategori": categories,
            "Optimal Ağırlık (%)": (optimal_weights.values * 100).round(2),
            f"Yatırılacak Tutar ({curr_symbol})": (
                optimal_weights.values * budget
            ).round(2),
        }
    )

    st.dataframe(
        alloc_df.style.format(
            {
                "Optimal Ağırlık (%)": "%{:.2f}",
                f"Yatırılacak Tutar ({curr_symbol})": f"{curr_symbol}{'{:,.2f}'}",
            }
        ),
        use_container_width=True,
    )

with col_right:
    fig_donut = px.pie(
        alloc_df,
        values="Optimal Ağırlık (%)",
        names="Ticker",
        hole=0.4,
        title="Varlık Dağılım Pastası",
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_donut, use_container_width=True)

# 3. Varlık Bazlı Detay Tablosu
st.subheader("📋 3. Varlık Bazlı Detay Tablosu")
asset_details = []
for ticker in returns_df.columns:
    ann_ret = returns_df[ticker].mean() * 252
    ann_vol = returns_df[ticker].std() * np.sqrt(252)
    w = optimal_weights[ticker]
    asset_details.append(
        {
            "Ticker": ticker,
            "Kategori": alloc_df.loc[
                alloc_df["Ticker"] == ticker, "Kategori"
            ].values[0],
            "Yıllık Beklenen Getiri (%)": round(ann_ret * 100, 2),
            "Bireysel Volatilite (Risk %)": round(ann_vol * 100, 2),
            "Optimal Ağırlık (%)": round(w * 100, 2),
            f"Yatırılacak Tutar ({curr_symbol})": round(w * budget, 2),
        }
    )

asset_df = pd.DataFrame(asset_details)
st.dataframe(asset_df, use_container_width=True)

st.divider()

# ==========================================
# 4. KORELASYON VE KOVARYANS MATRİSLERİ
# ==========================================
st.subheader("🧮 4. Korelasyon & Kovaryans Matrisleri")

tab_corr, tab_cov, tab_pinv, tab_ewma = st.tabs(
    [
        "🔗 Korelasyon Matrisi",
        "📐 Kovaryans Matrisi",
        "🔄 Ters Kovaryans Matrisi (Pinv)",
        "⚡ EWMA Dinamik Kovaryans",
    ]
)

with tab_corr:
    corr_matrix = returns_df.corr().round(2)
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_cov:
    st.dataframe(sample_cov.round(4), use_container_width=True)

with tab_pinv:
    st.write(
        "**Analitik Sharpe Optimizasyonunda kullanılan $Pinv(\\Sigma)$ Matrisi:**"
    )
    pinv_matrix = pd.DataFrame(
        np.linalg.pinv(sample_cov.values),
        index=sample_cov.index,
        columns=sample_cov.columns,
    )
    st.dataframe(pinv_matrix.round(4), use_container_width=True)

with tab_ewma:
    st.write(
        "**EWMA ($\lambda=0.94$) Dinamik Kovaryans Matrisi (Son günlere yüksek ağırlık verir):**"
    )
    ewma_cov = calculate_ewma_cov(returns_df)
    st.dataframe(ewma_cov.round(4), use_container_width=True)

st.divider()

# ==========================================
# 5. İLERİ KANTİTATİF ANALİTİK & SİMÜLASYONLAR
# ==========================================
st.subheader("📊 5. İleri Kantitatif Risk & Simülasyon Motoru")

q_tab1, q_tab2, q_tab3, q_tab4 = st.tabs(
    [
        "⚠️ Risk Metrikleri (VaR/CVaR/Kurtosis)",
        "🎲 Monte Carlo GBM (Cholesky)",
        "📈 GARCH(1,1) Volatilite",
        "🏛️ OU / Vasicek Faiz Simülasyonu",
    ]
)

# SEKMELER 1: Risk Metrikleri
with q_tab1:
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric(
        "Daily Value at Risk (%95 VaR)",
        f"%{metrics['VaR_95_Daily']*100:.2f}",
        help="1 günde kaybedilebilecek maks % tutar",
    )
    r_col2.metric(
        "Daily Expected Shortfall (%95 CVaR)",
        f"%{metrics['CVaR_95_Daily']*100:.2f}",
        help="Felaket senaryosunda beklenen ortalama kayıp",
    )
    r_col3.metric(
        "Skewness (Çarpıklık)",
        f"{metrics['Skewness']:.2f}",
        help="Negatif ise sol kuyruk (ani düşüş) riski yüksektir.",
    )
    r_col4.metric(
        "Excess Kurtosis (Basıklık)",
        f"{metrics['Kurtosis']:.2f}",
        help=">0 ise ekstrem tavan/taban günleri sık yaşanır.",
    )

    # Düşüş Dağılım Grafiği
    fig_hist = px.histogram(
        metrics["Daily_Returns"],
        nbins=50,
        title="Portföy Günlük Getiri Dağılımı ve Risk Sınırları (VaR)",
    )
    fig_hist.add_vline(
        x=-metrics["VaR_95_Daily"],
        line_dash="dash",
        line_color="red",
        annotation_text="%95 VaR Sınırı",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# SEKMELER 2: Monte Carlo Simulation
with q_tab2:
    st.write(
        "**Cholesky Ayrıştırması** ile varlıkların korelasyon yapısı korunarak **1000 Farklı Gelecek Patikası (1 Yıl / 252 Gün)** simüle edilmiştir."
    )
    mc_paths = run_monte_carlo_gbm(
        returns_df, optimal_weights, budget, days=252, N=500
    )

    fig_mc = go.Figure()
    # İlk 50 patikayı çizdir (Kasma olmaması için)
    for i in range(min(50, mc_paths.shape[1])):
        fig_mc.add_trace(
            go.Scatter(
                y=mc_paths[:, i],
                mode="lines",
                line=dict(width=1),
                showlegend=False,
                opacity=0.3,
            )
        )

    # Ortalama ve Alt/Üst Bantlar
    mean_path = np.mean(mc_paths, axis=1)
    p05_path = np.percentile(mc_paths, 5, axis=1)
    p95_path = np.percentile(mc_paths, 95, axis=1)

    fig_mc.add_trace(
        go.Scatter(
            y=mean_path,
            mode="lines",
            name="Ortalama Beklenti",
            line=dict(color="blue", width=3),
        )
    )
    fig_mc.add_trace(
        go.Scatter(
            y=p95_path,
            mode="lines",
            name="%95 İyimser Senaryo",
            line=dict(color="green", width=2, dash="dash"),
        )
    )
    fig_mc.add_trace(
        go.Scatter(
            y=p05_path,
            mode="lines",
            name="%5 Kötümser Senaryo",
            line=dict(color="red", width=2, dash="dash"),
        )
    )

    fig_mc.update_layout(
        title="Monte Carlo Gelecek Bütçe Simülasyonu",
        xaxis_title="İşlem Günü",
        yaxis_title=f"Portföy Değeri ({curr_symbol})",
    )
    st.plotly_chart(fig_mc, use_container_width=True)

# SEKMELER 3: GARCH(1,1) Volatilite
with q_tab3:
    st.write(
        "**GARCH(1,1) Zamanla Değişen Volatilite Tahmini vs Klasik Volatilite:**"
    )
    garch_results = []
    for t in returns_df.columns:
        hist_vol = returns_df[t].std() * np.sqrt(252)
        garch_vol = run_garch_volatility(returns_df[t])
        garch_results.append(
            {
                "Ticker": t,
                "Tarihsel Volatilite (%)": round(hist_vol * 100, 2),
                "GARCH(1,1) İleri Volatilite (%)": round(garch_vol * 100, 2),
                "Fark (Puan)": round((garch_vol - hist_vol) * 100, 2),
            }
        )

    st.dataframe(pd.DataFrame(garch_results), use_container_width=True)

# SEKMELER 4: Ornstein-Uhlenbeck / Vasicek
with q_tab4:
    st.write(
        "**Ornstein-Uhlenbeck / Vasicek Mean-Reverting Faiz Simülasyonu:**"
    )
    st.write(
        "Mevcut faiz oranının zamanla hedeflenen uzun dönemli ortalamaya ($dt$) nasıl yakınsadığını simüle eder."
    )

    ou_paths = run_ornstein_uhlenbeck(
        initial_rate=rf_rate_annual,
        target_rate=0.30,
        theta=0.15,
        sigma=0.04,
        days=252,
        N=100,
    )

    fig_ou = go.Figure()
    for i in range(30):
        fig_ou.add_trace(
            go.Scatter(
                y=ou_paths[:, i] * 100,
                mode="lines",
                opacity=0.2,
                showlegend=False,
            )
        )

    fig_ou.add_trace(
        go.Scatter(
            y=np.mean(ou_paths, axis=1) * 100,
            mode="lines",
            name="Ortalama Faiz Patikası",
            line=dict(color="orange", width=3),
        )
    )
    fig_ou.update_layout(
        title="Yıllık Politika/Mevduat Faizi Monte Carlo Patikası (OU Process)",
        xaxis_title="Gün",
        yaxis_title="Faiz Oranı (%)",
    )
    st.plotly_chart(fig_ou, use_container_width=True)
