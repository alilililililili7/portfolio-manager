from io import BytesIO
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
    page_title="Gelişmiş Multi-Currency Quant Portföy Engine",
    layout="wide",
    page_icon="🏛️",
)

# ==========================================
# 1. YARDIMCI VE KANTİTATİF FONKSİYONLAR
# ==========================================


def fetch_and_convert_data(tickers, base_currency="TRY (₺)", period="3y"):
    """Kullanıcının seçtiği baz para birimine göre (TL veya USD) verileri çeker ve dönüştürür."""
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
    return returns, converted_data


def calculate_ewma_cov(returns_df, lambda_param=0.94):
    """Dinamik EWMA Kovaryans Matrisi."""
    T, N = returns_df.shape
    returns = returns_df.values
    norm_returns = returns - np.mean(returns, axis=0)

    ewma_cov = np.zeros((N, N))
    for t in range(T):
        weight = (1 - lambda_param) * (lambda_param ** (T - 1 - t))
        ewma_cov += weight * np.outer(norm_returns[t], norm_returns[t])

    return ewma_cov * 252


def run_garch_volatility(returns_series):
    """GARCH(1,1) İleri Volatilite Tahmini."""
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


def build_hybrid_cov_matrix(returns_df):
    """GARCH(1,1) İleri Volatiliteler + EWMA Dinamik Korelasyon Matrisi."""
    ewma_cov = calculate_ewma_cov(returns_df)
    vols = np.sqrt(np.diag(ewma_cov))
    vols[vols == 0] = 1e-6
    corr_matrix = ewma_cov / np.outer(vols, vols)

    garch_vols = [
        run_garch_volatility(returns_df[col]) for col in returns_df.columns
    ]
    D_garch = np.diag(garch_vols)

    hybrid_cov = D_garch @ corr_matrix @ D_garch
    return pd.DataFrame(
        hybrid_cov, index=returns_df.columns, columns=returns_df.columns
    )


def run_ornstein_uhlenbeck(
    initial_rate, target_rate=None, theta=0.15, sigma=0.01, days=252, N=50
):
    """Ornstein-Uhlenbeck / Vasicek Faiz Görselleştirme Simülasyonu."""
    if target_rate is None:
        target_rate = initial_rate

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


def apply_ebitda_margin_penalty(
    expected_returns, ebitda_margins, min_threshold=0.15, max_penalty=0.40
):
    """EBITDA marjı düşük hisselerin getirisini matematiksel olarak cezalandırır/azaltır."""
    adjusted_returns = expected_returns.copy()
    for ticker, mu in expected_returns.items():
        margin = ebitda_margins.get(ticker, 0.20)
        if margin < min_threshold:
            penalty_ratio = (min_threshold - margin) / min_threshold
            penalty = min(penalty_ratio * max_penalty, max_penalty)
            adjusted_returns[ticker] = mu * (1.0 - penalty)
    return adjusted_returns


def calculate_portfolio_beta(weights, asset_betas):
    """Portföyün toplam piyasa duyarlılığını (Beta) hesaplar."""
    return sum(
        weights[ticker] * asset_betas.get(ticker, 1.0) for ticker in weights.index
    )


def calculate_fractional_kelly(
    portfolio_return, portfolio_vol, risk_free_rate, fraction=0.5
):
    """Fractional Kelly Kriteri ile optimal pozisyon büyüklüğü."""
    if portfolio_vol == 0:
        return 1.0
    full_kelly = (portfolio_return - risk_free_rate) / (portfolio_vol**2)
    return float(np.clip(full_kelly * fraction, 0.10, 1.0))


def apply_black_litterman_views(
    prior_returns, cov_matrix, views_dict, tau=0.05
):
    """Black-Litterman Modeli ile Yatırımcı Görüşlerini Beklenen Getirilere Entegre Eder."""
    if not views_dict:
        return prior_returns

    n = len(prior_returns)
    P = np.zeros((len(views_dict), n))
    Q = np.zeros(len(views_dict))

    for idx, (ticker, view_val) in enumerate(views_dict.items()):
        if ticker in prior_returns.index:
            col_idx = list(prior_returns.index).get_loc(
                ticker
            ) if hasattr(list(prior_returns.index), "get_loc") else list(prior_returns.index).index(ticker)
            P[idx, col_idx] = 1.0
            Q[idx] = view_val

    # Omega (Görüş Belirsizliği)
    omega = np.diag(np.diag(P @ (tau * cov_matrix.values) @ P.T)) * 1e-4

    # Black-Litterman Formülü
    inv_cov = pinv(tau * cov_matrix.values)
    inv_omega = pinv(omega) if omega.size > 0 else np.zeros((len(Q), len(Q)))

    middle_inv = pinv(inv_cov + P.T @ inv_omega @ P)
    bl_returns = middle_inv @ (
        inv_cov @ prior_returns.values + P.T @ inv_omega @ Q
    )

    return pd.Series(bl_returns, index=prior_returns.index)


def calculate_master_integrated_opt(
    returns_df,
    base_rf=0.45,
    ebitda_margins=None,
    dcf_potentials=None,
    max_asset_cap=0.25,
    max_bist_cap=0.50,
    bl_views=None,
):
    """GARCH + EWMA + Mean-CVaR + Birebir Rf Kullanımı + EBITDA Penalty + DCF Blend + Black-Litterman."""
    raw_mean_returns = returns_df.mean() * 252
    num_assets = len(raw_mean_returns)

    if ebitda_margins is None:
        ebitda_margins = {t: 0.20 for t in returns_df.columns}
    if dcf_potentials is None:
        dcf_potentials = raw_mean_returns.to_dict()

    effective_rf = float(base_rf)

    ou_sim = run_ornstein_uhlenbeck(
        initial_rate=base_rf, target_rate=base_rf, days=252, N=50
    )

    # GARCH + EWMA Hibrit Kovaryans Matrisi
    hybrid_cov_df = build_hybrid_cov_matrix(returns_df)
    lambda_ridge = 1e-4
    cov_values = hybrid_cov_df.values + np.eye(num_assets) * lambda_ridge

    # DCF + Quant Harmanlama
    blended_returns = {}
    for t in returns_df.columns:
        quant_ret = raw_mean_returns[t]
        dcf_pot = dcf_potentials.get(t, quant_ret)
        blended_returns[t] = (quant_ret * 0.70) + (dcf_pot * 0.30)
    blended_series = pd.Series(blended_returns)

    # Black-Litterman Görüşleri Uygulama
    if bl_views:
        blended_series = apply_black_litterman_views(
            blended_series, hybrid_cov_df, bl_views
        )

    # EBITDA Marjına Göre Getiriyi Azaltma
    adjusted_returns_dict = apply_ebitda_margin_penalty(
        blended_series, ebitda_margins
    )
    adjusted_returns = pd.Series(adjusted_returns_dict)

    # Kısıtlar
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    bist_indices = [
        i for i, col in enumerate(returns_df.columns) if col.endswith(".IS")
    ]
    if bist_indices and max_bist_cap < 1.0:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w, idx=bist_indices, max_w=max_bist_cap: max_w
                - np.sum(w[idx]),
            }
        )

    bounds = tuple((0.0, float(max_asset_cap)) for _ in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets

    def master_objective(weights):
        p_ret = np.dot(weights, adjusted_returns.values)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_values, weights)))

        port_daily = returns_df.dot(weights)
        var_95 = np.percentile(port_daily, 5)
        cvar_tail = abs(port_daily[port_daily <= var_95].mean())

        if p_vol == 0 or cvar_tail == 0:
            return 1e6

        sharpe = (p_ret - effective_rf) / p_vol
        cvar_penalty = cvar_tail * np.sqrt(252)

        return -(sharpe - 0.5 * cvar_penalty)

    opt_res = minimize(
        master_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    opt_w = np.maximum(opt_res.x, 0.0)
    opt_w = opt_w / np.sum(opt_w)

    return (
        pd.Series(opt_w, index=returns_df.columns),
        hybrid_cov_df,
        effective_rf,
        adjusted_returns,
        ou_sim,
    )


def calculate_advanced_metrics(
    returns_df, weights, adjusted_returns, risk_free_rate=0.45
):
    """VaR, CVaR, Skewness, Kurtosis ve Sharpe analizi."""
    port_daily_returns = returns_df.dot(weights)

    skewness = stats.skew(port_daily_returns)
    kurtosis = stats.kurtosis(port_daily_returns)

    var_95 = np.percentile(port_daily_returns, 5)
    cvar_tail = port_daily_returns[port_daily_returns <= var_95]
    cvar_95 = cvar_tail.mean() if len(cvar_tail) > 0 else var_95

    ann_return = np.dot(weights, adjusted_returns.values)
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
# 2. SIDEBAR (VARLIK VE AYARLAR)
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

st.sidebar.title("🏛️ Risksiz Faiz Ayarları")

col_rf1, col_rf2 = st.sidebar.columns(2)
rf_tl_input = (
    col_rf1.number_input("TL Faiz / Mevduat (%)", value=45.0, step=1.0) / 100.0
)
rf_usd_input = (
    col_rf2.number_input("USD Faiz / SOFR (%)", value=3.75, step=0.25) / 100.0
)

base_rf = rf_tl_input if "TRY" in base_currency else rf_usd_input

st.sidebar.title("🎯 Varlık Kısıtlamaları")
max_asset_cap = (
    st.sidebar.slider(
        "Maks. Tekil Varlık Limiti (%)",
        min_value=10,
        max_value=100,
        value=25,
        step=5,
    )
    / 100.0
)

max_bist_cap = (
    st.sidebar.slider(
        "Maks. BİST Toplam Limiti (%)",
        min_value=10,
        max_value=100,
        value=50,
        step=5,
    )
    / 100.0
)

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

use_realty = st.sidebar.checkbox("🏢 Emlak / Gayrimenkul (TL)", value=True)
realty_tickers_input = st.sidebar.text_input(
    "Emlak Hisseleri / GYO", "EKGYO.IS, SNGYO.IS"
)

use_sukuk = st.sidebar.checkbox(
    "📜 Kira Sertifikası / Sukuk (TL)", value=True
)
sukuk_tickers_input = st.sidebar.text_input(
    "Kira Sertifikası / Katılım", "KUVVA.IS"
)

selected_tickers = []
if use_stocks and stock_tickers_input:
    selected_tickers.extend(
        [t.strip() for t in stock_tickers_input.split(",") if t.strip()]
    )
if use_etfs and etf_tickers_input:
    selected_tickers.extend(
        [t.strip() for t in etf_tickers_input.split(",") if t.strip()]
    )
if use_commodities and commodity_tickers_input:
    selected_tickers.extend(
        [t.strip() for t in commodity_tickers_input.split(",") if t.strip()]
    )
if use_realty and realty_tickers_input:
    selected_tickers.extend(
        [t.strip() for t in realty_tickers_input.split(",") if t.strip()]
    )
if use_sukuk and sukuk_tickers_input:
    selected_tickers.extend(
        [t.strip() for t in sukuk_tickers_input.split(",") if t.strip()]
    )

# Yeni Özellik: Black-Litterman Yatırımcı Görüşleri Girişi
st.sidebar.title("🧠 Black-Litterman Görüşleri")
use_bl = st.sidebar.checkbox("Özel Getiri Beklentisi Tanımla", value=False)
bl_views = {}
if use_bl:
    view_ticker = st.sidebar.selectbox(
        "Hedef Varlık", selected_tickers if selected_tickers else [""]
    )
    view_expected_return = st.sidebar.number_input(
        "Öngörülen Yıllık Getiri (%)", value=30.0, step=5.0
    )
    if view_ticker:
        bl_views[view_ticker] = view_expected_return / 100.0

st.sidebar.divider()
run_opt_btn = st.sidebar.button(
    "🚀 Portföyü Optimize Et", type="primary", use_container_width=True
)

# ==========================================
# 3. ANA SAYFA VE HESAPLAMA MOTORU
# ==========================================
st.title("🏛️ Entegre Kantitatif Portföy Optimizasyon Engine")

if len(selected_tickers) < 2:
    st.warning("Lütfen optimizasyon için en az 2 varlık seçin.")
    st.stop()

returns_df, raw_price_df = fetch_and_convert_data(
    selected_tickers, base_currency=base_currency
)

if returns_df.empty or len(returns_df.columns) < 2:
    st.error(
        "Veri çekilemedi veya seçilen ticker'lar yetersiz. Lütfen ticker'ları kontrol edin."
    )
    st.stop()

# EBITDA, DCF & Beta Matrisleri
ebitda_margins = {
    t: 0.08 if "BIMAS" in t else (0.12 if "EREGL" in t else 0.25)
    for t in returns_df.columns
}
dcf_potentials = (returns_df.mean() * 252 * 1.1).to_dict()  # %10 DCF primi
asset_betas = {
    t: (0.85 if t in ["SPY", "VOO"] else (1.20 if t.endswith(".IS") else 0.60))
    for t in returns_df.columns
}

# MASTER OPTİMİZASYON
(
    optimal_weights,
    hybrid_cov_df,
    effective_rf,
    adjusted_returns,
    ou_sim_paths,
) = calculate_master_integrated_opt(
    returns_df,
    base_rf=base_rf,
    ebitda_margins=ebitda_margins,
    dcf_potentials=dcf_potentials,
    max_asset_cap=max_asset_cap,
    max_bist_cap=max_bist_cap,
    bl_views=bl_views,
)

metrics = calculate_advanced_metrics(
    returns_df,
    optimal_weights,
    adjusted_returns,
    risk_free_rate=effective_rf,
)

port_beta = calculate_portfolio_beta(optimal_weights, asset_betas)
kelly_size = calculate_fractional_kelly(
    metrics["Ann_Return"], metrics["Ann_Vol"], effective_rf, fraction=0.5
)

st.info(
    f"🌐 **Aktif Baz Para Birimi:** **{base_currency.split(' ')[0]}** | "
    f"**Model:** Entegre Quantamental (GARCH + EWMA + Mean-CVaR + EBITDA Penalty + DCF + BL) | "
    f"**$R_f$:** **%{effective_rf*100:.2f}** | "
    f"**Tek Varlık / BİST Sınırı:** %{max_asset_cap*100:.0f} / %{max_bist_cap*100:.0f}"
)

# ==========================================
# 1. GENİŞ PORTFÖY ÖZETLERİ & METRİKLER
# ==========================================
st.subheader(f"📌 1. Optimum Portföy Genel Özeti ({curr_symbol} Bazlı)")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Toplam Bütçe", f"{curr_symbol}{budget:,.2f}")
col2.metric("Beklenen Yıllık Getiri", f"%{metrics['Ann_Return']*100:.2f}")
col3.metric("Portföy Volatilite (Risk)", f"%{metrics['Ann_Vol']*100:.2f}")
col4.metric("Sharpe Oranı", f"{metrics['Sharpe']:.2f}")
col5.metric("Portföy Betası", f"{port_beta:.2f}")

# Kelly Position Size Görselleştirme
st.markdown("##### 🎯 Dynamic Position Size (Kelly Kriteri & Nakit Tamponu)")
k_col1, k_col2 = st.columns([1.5, 2.5])
with k_col1:
    st.write(
        f"**Hissede Kalınacak Tutar:** %{kelly_size*100:.1f} ({curr_symbol}{budget*kelly_size:,.2f})"
    )
    st.write(
        f"**Nakit/Mevduat Tamponu:** %{(1-kelly_size)*100:.1f} ({curr_symbol}{budget*(1-kelly_size):,.2f})"
    )
with k_col2:
    st.progress(kelly_size)

st.divider()

# ==========================================
# 2. SEKTÖR / KATEGORİ VE VARLIK TABLOSU
# ==========================================
col_left, col_right = st.columns([1.3, 1])

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
                optimal_weights.values * (budget * kelly_size)
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
        title="Optimum Varlık Dağılım Pastası",
    )
    fig_donut.update_traces(textposition="inside", textinfo="percent+label")
    st.plotly_chart(fig_donut, use_container_width=True)

# 3. Varlık Bazlı Detay Tablosu (EBITDA & Beta Dahil)
st.subheader(
    f"📋 3. Varlık Bazlı Detay Tablosu ({base_currency.split(' ')[0]} Bazında)"
)
asset_details = []
for ticker in returns_df.columns:
    ann_ret = adjusted_returns[ticker]
    ann_vol = returns_df[ticker].std() * np.sqrt(252)
    w = optimal_weights[ticker]
    asset_details.append(
        {
            "Ticker": ticker,
            "Kategori": alloc_df.loc[
                alloc_df["Ticker"] == ticker, "Kategori"
            ].values[0],
            "EBITDA Marjı (%)": round(ebitda_margins.get(ticker, 0.20) * 100, 1),
            "Beta": round(asset_betas.get(ticker, 1.0), 2),
            f"Düzeltilmiş $µ$ Getiri ({curr_symbol} %)": round(ann_ret * 100, 2),
            "Bireysel Volatilite (Risk %)": round(ann_vol * 100, 2),
            "Optimal Ağırlık (%)": round(w * 100, 2),
            f"Yatırılacak Tutar ({curr_symbol})": round(
                w * (budget * kelly_size), 2
            ),
        }
    )

asset_df = pd.DataFrame(asset_details)
st.dataframe(asset_df, use_container_width=True)

st.divider()

# ==========================================
# 4. KORELASYON VE KOVARYANS MATRİSLERİ
# ==========================================
st.subheader("🧮 4. Korelasyon & Hibrit Kovaryans Matrisi")

tab_corr, tab_cov, tab_pinv = st.tabs(
    [
        "🔗 Korelasyon Matrisi",
        "📐 GARCH+EWMA Hibrit Kovaryans Matrisi",
        "🔄 Ters Kovaryans Matrisi (Pinv)",
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
    st.dataframe(hybrid_cov_df.round(4), use_container_width=True)

with tab_pinv:
    st.write(
        "**Ridge Regülarize Hibrit Kovaryans Matrisinin Tersi ($Pinv(\\Sigma_{hybrid} + \\lambda I)$):**"
    )
    pinv_matrix = pd.DataFrame(
        pinv(hybrid_cov_df.values),
        index=hybrid_cov_df.index,
        columns=hybrid_cov_df.columns,
    )
    st.dataframe(pinv_matrix.round(4), use_container_width=True)

st.divider()

# ==========================================
# 5. İLERİ KANTİTATİF ANALİTİK & SİMÜLASYONLAR
# ==========================================
st.subheader("📊 5. İleri Kantitatif Risk & Simülasyon Motoru")

q_tab1, q_tab2, q_tab3, q_tab4, q_tab5, q_tab6 = st.tabs(
    [
        "⚠️ Risk Metrikleri (VaR/CVaR/Kurtosis)",
        "🎲 Monte Carlo GBM (Cholesky)",
        "📈 GARCH(1,1) Volatilite",
        "🏛️ OU / Vasicek Faiz Simülasyonu",
        "🔙 Geçmişe Dönük Backtest",
        "⚡ Kriz Stres Testi",
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
        returns_df, optimal_weights, budget * kelly_size, days=252, N=500
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
        "**Ornstein-Uhlenbeck / Vasicek Mean-Reverting Faiz Görselleştirme Simülasyonu:**"
    )

    ou_paths = run_ornstein_uhlenbeck(
        initial_rate=base_rf,
        target_rate=base_rf,
        theta=0.15,
        sigma=0.01 if "USD" in base_currency else 0.03,
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

# YENİ EKLENEN SEKME 5: Backtest Analizi
with q_tab5:
    st.subheader("🔙 Geçmişe Dönük Portföy Performans Backtesti")
    st.write(
        "Optimize edilen ağırlıkların geçmiş veriler üzerindeki kümülatif büyüme seyri ve Benchmark (Örn: SPY / XU100) kıyaslaması."
    )

    portfolio_cum_returns = (1 + returns_df.dot(optimal_weights)).cumprod() * (
        budget * kelly_size
    )

    fig_backtest = go.Figure()
    fig_backtest.add_trace(
        go.Scatter(
            x=portfolio_cum_returns.index,
            y=portfolio_cum_returns.values,
            mode="lines",
            name="Quant Optimizasyon Portföyü",
            line=dict(color="green", width=2.5),
        )
    )

    # Benchmark ekleme (Eğer SPY veya başka bir varlık varsa)
    benchmark_ticker = "SPY" if "SPY" in raw_price_df.columns else raw_price_df.columns[0]
    if benchmark_ticker in raw_price_df.columns:
        bench_cum = (
            raw_price_df[benchmark_ticker]
            / raw_price_df[benchmark_ticker].iloc[0]
        ) * (budget * kelly_size)
        fig_backtest.add_trace(
            go.Scatter(
                x=bench_cum.index,
                y=bench_cum.values,
                mode="lines",
                name=f"Benchmark ({benchmark_ticker})",
                line=dict(color="gray", width=1.5, dash="dash"),
            )
        )

    fig_backtest.update_layout(
        title="Kümülatif Portföy Getiri Eğrisi",
        xaxis_title="Tarih",
        yaxis_title=f"Portföy Değeri ({curr_symbol})",
    )
    st.plotly_chart(fig_backtest, use_container_width=True)

# YENİ EKLENEN SEKME 6: Stres Testi (Kriz Senaryoları)
with q_tab6:
    st.subheader("⚡ Portföy Kriz Stres Testi (Stress Testing)")
    st.write(
        "Piyasada ani şoklar (Örn: Borsa Çöküşü, Kur Şoku, Savaş veya Likidite Krizi) yaşanması durumunda portföyün maruz kalacağı tahmini kayıp."
    )

    col_s1, col_s2, col_s3 = st.columns(3)

    # Senaryo 1: Genel Piyasa %20 Düşüşü
    shock_market = -0.20
    portfolio_shock_loss = shock_market * port_beta * (budget * kelly_size)

    # Senaryo 2: Faizlerin Ani 500 Baz Puan Artışı
    shock_rate = -0.08 * (budget * kelly_size)

    # Senaryo 3: Döviz / Kur Krizi (Eğer USD bazlı varlıklar varsa pozitif etkilenme)
    shock_fx = (
        0.15 * sum(optimal_weights[t] for t in returns_df.columns if not t.endswith(".IS"))
        * (budget * kelly_size)
    )

    col_s1.metric(
        "Şok Senaryosu: Global Crash (%20)",
        f"{curr_symbol}{portfolio_shock_loss:,.2f}",
        delta=f"%{shock_market*port_beta*100:.1f}",
        delta_color="inverse",
    )
    col_s2.metric(
        "Şok Senaryosu: Faiz Şoku (+500 bps)",
        f"{curr_symbol}{shock_rate:,.2f}",
        delta="% -8.0",
        delta_color="inverse",
    )
    col_s3.metric(
        "Şok Senaryosu: Kur / FX Şoku",
        f"{curr_symbol}{shock_fx:,.2f}",
        delta=f"+%{shock_fx/(budget*kelly_size)*100:.1f}",
    )

st.divider()

# ==========================================
# 6. RAPOR DIŞA AKTARIMI (CSV & EXCEL)
# ==========================================
st.subheader("📥 Raporu Dışa Aktar")

col_dl1, col_dl2 = st.columns(2)

with col_dl1:
    csv_data = asset_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📄 Varlık Dağılımını CSV Olarak İndir",
        data=csv_data,
        file_name="quant_portfoy_dagilimi.csv",
        mime="text/csv",
        use_container_width=True,
    )

with col_dl2:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        asset_df.to_excel(writer, sheet_name="Optimum_Dagilim", index=False)
        alloc_df.to_excel(writer, sheet_name="Kategoriler", index=False)
    excel_data = output.getvalue()
    st.download_button(
        label="📊 Detaylı Raporu Excel Olarak İndir",
        data=excel_data,
        file_name="quant_portfoy_detayli_rapor.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
