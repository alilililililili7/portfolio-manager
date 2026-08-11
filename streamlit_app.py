import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.linalg import pinv
from scipy.optimize import minimize
import scipy.stats as stats
import streamlit as st
import yfinance as yf

# ARCH/GARCH Modülü Güvenlik Kontrolü
try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

st.set_page_config(
    page_title="Gelişmiş Multi-Currency Portföy Engine",
    layout="wide",
    page_icon="🏛️",
)

# ==========================================
# 1. YARDIMCI VE KANTİTATİF FONKSİYONLAR
# ==========================================


def fetch_and_convert_data(tickers, base_currency="TRY (₺)", period="3y"):
    """
    Kullanıcının seçtiği baz para birimine göre (TL veya USD):
    - TL Seçildiyse: USD varlıkları USD/TRY kuruyla ÇARPAR.
    - USD Seçildiyse: TL varlıkları TRY/USD kuruyla ÇARPAR (USDTRY'ye BÖLER).
    """
    if not tickers:
        return pd.DataFrame()

    all_symbols = list(set(tickers + ["USDTRY=X"]))
    data = yf.download(all_symbols, period=period, progress=False)["Close"]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(how="all").ffill().bfill()

    # FX Kurunu Ayır
    usd_try = data["USDTRY=X"] if "USDTRY=X" in data.columns else None
    if "USDTRY=X" in data.columns:
        data = data.drop(columns=["USDTRY=X"])

    converted_data = pd.DataFrame(index=data.index)

    for col in data.columns:
        is_bist_asset = col.endswith(".IS")

        if base_currency.startswith("TRY"):
            if is_bist_asset or usd_try is None:
                converted_data[col] = data[col]
            else:
                converted_data[col] = data[col] * usd_try
        else:
            if is_bist_asset and usd_try is not None:
                converted_data[col] = data[col] / usd_try
            else:
                converted_data[col] = data[col]

    returns = (
        converted_data.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    )
    return returns


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
        return (np.sqrt(forecast_var) / 100) * np.sqrt(252)
    except Exception:
        return returns_series.std() * np.sqrt(252)


def build_garch_cov_matrix(returns_df):
    """GARCH ileri volatiliteleri ile Korelasyon matrisinden Kovaryans matrisi üretir."""
    corr = returns_df.corr().values
    garch_vols = [run_garch_volatility(returns_df[col]) for col in returns_df.columns]
    D = np.diag(garch_vols)
    garch_cov = D @ corr @ D
    return pd.DataFrame(garch_cov, index=returns_df.columns, columns=returns_df.columns)


def run_ornstein_uhlenbeck(initial_rate, target_rate=0.35, theta=0.15, sigma=0.04, days=252, N=100):
    """Faiz Oranı Mean-Reversion Simülasyonu."""
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


def calculate_advanced_portfolio_opt(
    returns_df,
    risk_free_rate=0.45,
    method="Klasik Sharpe (MPT)",
    max_asset_weight=1.0,
    max_bist_weight=1.0,
    use_ewma=False,
):
    """Gelişmiş Optimizasyon Engine: Sharpe, GARCH-Sharpe, Mean-CVaR, Risk Parity ve Kısıtlamalar."""
    mean_returns = returns_df.mean() * 252
    num_assets = len(mean_returns)

    # Kovaryans Matrisi Seçimi
    if method == "GARCH-Sharpe":
        cov_matrix = build_garch_cov_matrix(returns_df)
    elif use_ewma:
        cov_matrix = calculate_ewma_cov(returns_df)
    else:
        cov_matrix = returns_df.cov() * 252

    lambda_ridge = 1e-4
    cov_values = cov_matrix.values + np.eye(num_assets) * lambda_ridge

    # Kısıtlar (Constraints)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # BİST Toplam Limit Kısıtı
    bist_indices = [i for i, col in enumerate(returns_df.columns) if col.endswith(".IS")]
    if bist_indices and max_bist_weight < 1.0:
        constraints.append({
            "type": "ineq",
            "fun": lambda w, idx=bist_indices, max_w=max_bist_weight: max_w - np.sum(w[idx])
        })

    # Sınırlar (Bounds - Tekil Varlık Sınırı)
    bounds = tuple((0.0, float(max_asset_weight)) for _ in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets

    # A) Sharpe Optimizasyonu
    if method in ["Klasik Sharpe (MPT)", "GARCH-Sharpe"]:
        def negative_sharpe(weights):
            p_ret = np.dot(weights, mean_returns)
            p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_values, weights)))
            if p_vol == 0:
                return 1e6
            return -(p_ret - risk_free_rate) / p_vol

        opt_res = minimize(
            negative_sharpe,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        opt_w = opt_res.x

    # B) Mean-CVaR (Tail Risk) Optimizasyonu
    elif method == "Mean-CVaR (Tail Risk)":
        def negative_mean_cvar(weights):
            port_daily = returns_df.dot(weights)
            var_95 = np.percentile(port_daily, 5)
            cvar = abs(port_daily[port_daily <= var_95].mean())
            ann_ret = np.dot(weights, mean_returns)
            if cvar == 0:
                return 1e6
            return -(ann_ret - risk_free_rate) / (cvar * np.sqrt(252))

        opt_res = minimize(
            negative_mean_cvar,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        opt_w = opt_res.x

    # C) Risk Paritesi (Risk Parity)
    elif method == "Risk Paritesi (Risk Parity)":
        def risk_budget_objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_values, weights)))
            marginal_risk = np.dot(cov_values, weights) / portfolio_vol
            risk_contribution = weights * marginal_risk
            target_risk = portfolio_vol / num_assets
            return np.sum((risk_contribution - target_risk) ** 2)

        opt_res = minimize(
            risk_budget_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        opt_w = opt_res.x

    opt_w = np.maximum(opt_w, 0.0)
    opt_w = opt_w / np.sum(opt_w)

    cov_df = pd.DataFrame(cov_values, index=returns_df.columns, columns=returns_df.columns)
    return pd.Series(opt_w, index=returns_df.columns), cov_df


def calculate_advanced_metrics(returns_df, weights, risk_free_rate=0.45):
    """VaR, CVaR, Skewness, Kurtosis ve Sharpe analizi."""
    port_daily_returns = returns_df.dot(weights)

    skewness = stats.skew(port_daily_returns)
    kurtosis = stats.kurtosis(port_daily_returns)

    var_95 = np.percentile(port_daily_returns, 5)
    cvar_tail = port_daily_returns[port_daily_returns <= var_95]
    cvar_95 = cvar_tail.mean() if len(cvar_tail) > 0 else var_95

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


def run_monte_carlo_gbm(returns_df, weights, initial_budget, days=252, N=500):
    """Cholesky Ayrıştırması ile GBM Monte Carlo Simülasyonu."""
    port_returns = returns_df.dot(weights)
    mu = port_returns.mean()
    cov = returns_df.cov().values

    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov += np.eye(cov.shape[0]) * 1e-6
        L = np.linalg.cholesky(cov)

    num_assets = len(weights)
    w = weights.values

    simulation_paths = np.zeros((days, N))

    for i in range(N):
        Z = np.random.normal(size=(days, num_assets))
        correlated_Z = np.dot(Z, L.T)
        port_shocks = np.dot(correlated_Z, w)
        daily_factors = 1 + (mu + port_shocks)
        simulation_paths[:, i] = initial_budget * np.cumprod(daily_factors)

    return simulation_paths


# ==========================================
# 2. SIDEBAR (VARLIK, MODEL VE KISIT GİRDİLERİ)
# ==========================================
st.sidebar.title("💰 Baz Para Birimi & Bütçe")

base_currency = st.sidebar.selectbox(
    "Portföy Baz Para Birimi",
    ["TRY (₺) - TL Bazlı Raporlama", "USD ($) - Dolar Bazlı Raporlama"],
)

curr_symbol = "₺" if "TRY" in base_currency else "$"

budget = st.sidebar.number_input(
    f"Toplam Bütçe ({curr_symbol})",
    min_value=1000.0,
    value=100000.0,
    step=5000.0,
)

st.sidebar.title("🏛️ Dinamik Risksiz Faiz Ayarları")

col_rf1, col_rf2 = st.sidebar.columns(2)
rf_tl_input = (
    col_rf1.number_input("TL Faiz / Mevduat (%)", value=45.0, step=1.0) / 100.0
)
rf_usd_input = (
    col_rf2.number_input("USD Faiz / SOFR (%)", value=5.0, step=0.5) / 100.0
)

# Seçilen Baz Para Birimine Göre Aktif Risksiz Faiz
base_rf = rf_tl_input if "TRY" in base_currency else rf_usd_input

# Ornstein-Uhlenbeck (OU) Dinamik Faiz Hesabı Entegrasyonu
use_ou_rf = st.sidebar.checkbox("🔄 OU Mean-Reverting Dinamik Faiz Kullan", value=True)
if use_ou_rf:
    target_rf_sim = 0.35 if "TRY" in base_currency else 0.035
    ou_sim = run_ornstein_uhlenbeck(initial_rate=base_rf, target_rate=target_rf_sim, days=252, N=50)
    effective_rf = float(np.mean(ou_sim))
    st.sidebar.caption(f"💡 Simüle Edilen Dinamik $R_f$: **%{effective_rf*100:.2f}**")
else:
    effective_rf = base_rf

st.sidebar.title("🎯 Optimizasyon Modu & Kısıtlar")

opt_method = st.sidebar.selectbox(
    "Optimizasyon Yöntemi",
    [
        "Klasik Sharpe (MPT)",
        "GARCH-Sharpe",
        "Mean-CVaR (Tail Risk)",
        "Risk Paritesi (Risk Parity)",
    ],
)

use_ewma_cov = st.sidebar.checkbox("⚡ EWMA Dinamik Kovaryans Matrisi Kullan", value=False)

max_asset_cap = st.sidebar.slider(
    "Maks. Tekil Varlık Limiti (%)",
    min_value=10,
    max_value=100,
    value=25,
    step=5,
) / 100.0

max_bist_cap = st.sidebar.slider(
    "Maks. BİST Toplam Limiti (%)",
    min_value=10,
    max_value=100,
    value=50,
    step=5,
) / 100.0

st.sidebar.title("🏛️ Varlık Seçimleri")

use_stocks = st.sidebar.checkbox("📈 BİST Hisseleri (TL)", value=True)
stock_tickers_input = st.sidebar.text_input(
    "BİST Hisseleri",
    "THYAO.IS, EREGL.IS, KCHOL.IS, TUPRS.IS, BIMAS.IS, AKBNK.IS, SISE.IS",
)

use_etfs = st.sidebar.checkbox("🧺 ABD ETF / Hisseleri (USD)", value=True)
etf_tickers_input = st.sidebar.text_input("ABD Varlıkları", "SPY, VOO")

use_commodities = st.sidebar.checkbox("🥇 Emtialar (USD)", value=True)
commodity_tickers_input = st.sidebar.text_input("Emtialar", "IAU, CPER")

# YENİ EKLENEN: EMLAK VE KİRA SERTİFİKASI (SUKUK) CHECKBOX'LARI
use_realty = st.sidebar.checkbox("🏢 Emlak / Gayrimenkul (TL)", value=True)
realty_tickers_input = st.sidebar.text_input("Emlak Hisseleri / GYO", "EKGYO.IS, SNGYO.IS")

use_sukuk = st.sidebar.checkbox("📜 Kira Sertifikası / Sukuk (TL)", value=True)
sukuk_tickers_input = st.sidebar.text_input("Kira Sertifikası / Katılım", "KUVVA.IS")

selected_tickers = []
if use_stocks and stock_tickers_input:
    selected_tickers.extend([t.strip() for t in stock_tickers_input.split(",") if t.strip()])
if use_etfs and etf_tickers_input:
    selected_tickers.extend([t.strip() for t in etf_tickers_input.split(",") if t.strip()])
if use_commodities and commodity_tickers_input:
    selected_tickers.extend([t.strip() for t in commodity_tickers_input.split(",") if t.strip()])
if use_realty and realty_tickers_input:
    selected_tickers.extend([t.strip() for t in realty_tickers_input.split(",") if t.strip()])
if use_sukuk and sukuk_tickers_input:
    selected_tickers.extend([t.strip() for t in sukuk_tickers_input.split(",") if t.strip()])

st.sidebar.divider()
run_opt_btn = st.sidebar.button("🚀 Portföyü Optimize Et / Hesapla", type="primary", use_container_width=True)

# ==========================================
# 3. ANA SAYFA VE HESAPLAMA MOTORU
# ==========================================
st.title("🏛️ Simetrik Multi-Currency Portföy Optimizasyon Motoru")

if len(selected_tickers) < 2:
    st.warning("Lütfen optimizasyon için en az 2 varlık seçin.")
    st.stop()

# Dönüştürülmüş Getiri Verilerini Çekme
returns_df = fetch_and_convert_data(
    selected_tickers, base_currency=base_currency
)

if returns_df.empty or len(returns_df.columns) < 2:
    st.error(
        "Veri çekilemedi veya seçilen ticker'lar yetersiz. Lütfen ticker'ları kontrol edin."
    )
    st.stop()

# Optimizasyon ve Metrikler
optimal_weights, sample_cov = calculate_advanced_portfolio_opt(
    returns_df,
    risk_free_rate=effective_rf,
    method=opt_method,
    max_asset_weight=max_asset_cap,
    max_bist_weight=max_bist_cap,
    use_ewma=use_ewma_cov,
)

metrics = calculate_advanced_metrics(
    returns_df, optimal_weights, risk_free_rate=effective_rf
)

# Bilgi Çubuğu
conversion_info = (
    "USD varlıklar USD/TRY kuru ile ÇARPILARAK TL'ye dönüştürüldü."
    if "TRY" in base_currency
    else "BİST (TL) varlıklar TRY/USD kuru ile ÇARPILARAK (USDTRY'ye bölünerek) USD'ye dönüştürüldü."
)

st.info(
    f"🌐 **Aktif Baz Para Birimi:** **{base_currency.split(' ')[0]}** | "
    f"**Model Yöntemi:** `{opt_method}` | "
    f"**Modeldeki Risksiz Faiz ($R_f$):** **%{effective_rf*100:.2f}** | "
    f"**Maks Tek Varlık / BİST Sınırı:** %{max_asset_cap*100:.0f} / %{max_bist_cap*100:.0f}"
)

# ==========================================
# 1. GENİŞ PORTFÖY ÖZETLERİ
# ==========================================
st.subheader(f"📌 1. Portföy Genel Özeti ({curr_symbol} Bazlı)")

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
    st.subheader("📋 2. Sektör & Kategori Bazlı Dağılım")
    categories = []
    realty_list = [t.strip() for t in realty_tickers_input.split(",")]
    sukuk_list = [t.strip() for t in sukuk_tickers_input.split(",")]

    for ticker in optimal_weights.index:
        if ticker in realty_list or "EKGYO" in ticker or "GYO" in ticker:
            categories.append("Emlak / Gayrimenkul (TL)")
        elif ticker in sukuk_list:
            categories.append("Kira Sertifikası / Sukuk (TL)")
        elif ticker.endswith(".IS"):
            categories.append("Hisse (BİST / TL)")
        elif ticker in ["SPY", "VOO"]:
            categories.append("ETF (ABD / USD)")
        elif ticker in ["IAU", "CPER"]:
            categories.append("Emtia (USD)")
        else:
            categories.append("Diğer")

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
st.subheader(
    f"📋 3. Varlık Bazlı Detay Tablosu ({base_currency.split(' ')[0]} Bazında)"
)
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
            f"Yıllık Beklenen Getiri ({curr_symbol} %)": round(ann_ret * 100, 2),
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

tab_corr, tab_cov, tab_pinv, tab_ewma, tab_garch_cov = st.tabs(
    [
        "🔗 Korelasyon Matrisi",
        "📐 Kovaryans Matrisi (Aktif)",
        "🔄 Ters Kovaryans Matrisi (Pinv)",
        "⚡ EWMA Dinamik Kovaryans",
        "📈 GARCH İleri Kovaryans Matrisi",
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
        "**Ridge Regülarize Kovaryans Matrisinin Tersi ($Pinv(\\Sigma + \\lambda I)$):**"
    )
    pinv_matrix = pd.DataFrame(
        pinv(sample_cov.values),
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

with tab_garch_cov:
    st.write(
        "**GARCH(1,1) İleri Volatiliteler ile Oluşturulmuş Kovaryans Matrisi ($\Sigma_{GARCH}$):**"
    )
    garch_cov_df = build_garch_cov_matrix(returns_df)
    st.dataframe(garch_cov_df.round(4), use_container_width=True)

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
        "**Cholesky Ayrıştırması** ile varlıkların korelasyon yapısı korunarak **500 Farklı Gelecek Patikası (1 Yıl / 252 Gün)** simüle edilmiştir."
    )
    mc_paths = run_monte_carlo_gbm(
        returns_df, optimal_weights, budget, days=252, N=500
    )

    fig_mc = go.Figure()
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

    target_rate_sim = 0.30 if "TRY" in base_currency else 0.035

    ou_paths = run_ornstein_uhlenbeck(
        initial_rate=base_rf,
        target_rate=target_rate_sim,
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
        title=f"Yıllık {base_currency.split(' ')[0]} Risksiz Faiz Monte Carlo Patikası (OU Process)",
        xaxis_title="Gün",
        yaxis_title="Faiz Oranı (%)",
    )
    st.plotly_chart(fig_ou, use_container_width=True)
