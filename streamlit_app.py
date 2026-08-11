import sys
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import streamlit as st

# Yahoo Finance entegrasyonu (Gerçek veri çekimi için)
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. STREAMLIT CONFIGURATION & CUSTOM STYLES
# ==============================================================================

st.set_page_config(
    page_title="Quantamental Portfolio Optimization Terminal",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
</style>
""",
    unsafe_allow_allowed_html=True,
)

# ==============================================================================
# 2. GLOBAL CONFIGURATION & GLOSSARY
# ==============================================================================

GLOBAL_CONFIG = {
    "TRADING_DAYS_YEAR": 252,
    "CONFIDENCE_LEVEL_95": 0.95,
    "CONFIDENCE_LEVEL_99": 0.99,
    "DEFAULT_LAMBDA_RIDGE": 1e-4,
    "MAX_OPTIMIZATION_ITERATIONS": 1500,
    "TOLERANCE": 1e-9,
}

# ==============================================================================
# 3. INTERACTIVE SIDEBAR CONTROL PANEL
# ==============================================================================

st.sidebar.markdown("## ⚙️ Terminal Kontrol Paneli")
st.sidebar.markdown("---")

st.sidebar.subheader("📌 Varlık Seçimi")
default_assets = "THYAO.IS, GARAN.IS, ASELS.IS, KCHOL.IS, BIMAS.IS, AAPL, MSFT, NVDA, AMZN, GLD, VNQ"
user_ticker_input = st.sidebar.text_area(
    "Varlık Ticker'larını Girin (Virgülle Ayırın):",
    value=default_assets,
    height=100,
    help="Hisseler, ETF'ler (GLD, VNQ vb.) veya BIST hisseleri (.IS ekli)",
)

TARGET_TICKERS = [
    t.strip().upper() for t in user_ticker_input.split(",") if t.strip()
]

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Makro & Risk Parametreleri")

BASE_RF_RATE = (
    st.sidebar.slider(
        "Gözlemlenen Politika / Gösterge Faizi (%)",
        min_value=0.0,
        max_value=60.0,
        value=45.0,
        step=0.5,
    )
    / 100.0
)

MIN_ASSET_WEIGHT = (
    st.sidebar.slider(
        "Varlık Başı Minimum Taban Ağırlık (%)",
        min_value=0.0,
        max_value=10.0,
        value=1.5,
        step=0.5,
    )
    / 100.0
)

MAX_ASSET_WEIGHT = (
    st.sidebar.slider(
        "Varlık Başı Maksimum Tavan Ağırlık (%)",
        min_value=5.0,
        max_value=100.0,
        value=30.0,
        step=5.0,
    )
    / 100.0
)

MAX_BIST_CAP = (
    st.sidebar.slider(
        "Portföydeki Maksimum BIST Payı Tavanı (%)",
        min_value=10.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
    )
    / 100.0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Veri & Yöntem Tercihleri")

DATA_SOURCE_CHOICE = st.sidebar.radio(
    "Veri Kaynağı Modu",
    options=["Yahoo Finance (Canlı/Tarihsel)", "Sentetik / Monte Carlo Simülasyonu"],
    index=0 if YFINANCE_AVAILABLE else 1,
)

ESTIMATION_METHOD = st.sidebar.selectbox(
    "Kovaryans Tahmin Yöntemi",
    options=["Hibrit (EWMA + Sample Cov)", "Sample Covariance", "EWMA Covariance"],
    index=0,
)

# ==============================================================================
# 4. ADVANCED FINANCIAL DATA INGESTION ENGINE
# ==============================================================================


class FinancialDataIngestor:
    """Hisseler, ETF'ler ve Emtialar için veri çekme ve sentetik veri üretme modülü."""

    def __init__(self, tickers, start_date="2023-01-01", end_date="2026-08-11"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_real_data(self):
        if not YFINANCE_AVAILABLE:
            st.warning(
                "yfinance kütüphanesi ortamda yüklü değil. Sentetik veriye geçiliyor."
            )
            return self.generate_synthetic_returns()

        try:
            raw_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
            )
            if "Adj Close" in raw_data:
                price_df = raw_data["Adj Close"]
            elif "Close" in raw_data:
                price_df = raw_data["Close"]
            else:
                price_df = raw_data

            price_df = price_df.dropna(how="all").ffill().bfill()
            returns_df = price_df.pct_change().dropna()

            if isinstance(returns_df, pd.Series):
                returns_df = returns_df.to_frame(name=self.tickers[0])

            # Eksik kalan ticker'ları temizle
            valid_cols = [col for col in returns_df.columns if col in self.tickers]
            returns_df = returns_df[valid_cols]

            return returns_df
        except Exception as e:
            st.error(f"Yahoo Finance API Hatası: {e}. Sentetik moda geçiliyor.")
            return self.generate_synthetic_returns()

    def generate_synthetic_returns(self, seed=42):
        np.random.seed(seed)
        dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq="B"
        )
        num_days = len(dates)
        num_assets = len(self.tickers)

        drifts = np.random.uniform(0.0002, 0.0010, size=num_assets)
        vols = np.random.uniform(0.010, 0.025, size=num_assets)

        raw_cov = np.random.uniform(0.10, 0.50, size=(num_assets, num_assets))
        cov_matrix = (raw_cov + raw_cov.T) / 2
        np.fill_diagonal(cov_matrix, 1.0)

        L = np.linalg.cholesky(cov_matrix)
        uncorrelated_shocks = np.random.normal(
            0, 1, size=(num_days, num_assets)
        )
        correlated_shocks = np.dot(uncorrelated_shocks, L.T)

        daily_returns = drifts + correlated_shocks * vols
        returns_df = pd.DataFrame(
            daily_returns, index=dates, columns=self.tickers
        )

        return returns_df

    def fetch_fundamental_metrics(self):
        np.random.seed(101)
        pe_ratios = {}
        pb_ratios = {}
        ev_ebitda_ratios = {}
        ebit_growths = {}
        ebitda_growths = {}
        ebitda_margins = {}
        dcf_potentials = {}

        for t in self.tickers:
            is_bist = t.endswith(".IS")
            is_etf = t in ["GLD", "SLV", "VNQ", "QQQ", "SPY", "TLT", "SHY"]

            if is_etf:
                pe_ratios[t] = 0.0
                pb_ratios[t] = 0.0
                ev_ebitda_ratios[t] = 0.0
                ebit_growths[t] = 0.05
                ebitda_growths[t] = 0.05
                ebitda_margins[t] = 0.20
                dcf_potentials[t] = np.random.uniform(0.08, 0.25)
            elif is_bist:
                pe_ratios[t] = np.random.uniform(4.5, 18.0)
                pb_ratios[t] = np.random.uniform(0.8, 3.8)
                ev_ebitda_ratios[t] = np.random.uniform(3.5, 12.0)
                ebit_growths[t] = np.random.uniform(-0.15, 0.75)
                ebitda_growths[t] = np.random.uniform(-0.10, 0.65)
                ebitda_margins[t] = np.random.uniform(0.08, 0.42)
                dcf_potentials[t] = np.random.uniform(0.15, 0.70)
            else:
                pe_ratios[t] = np.random.uniform(14.0, 40.0)
                pb_ratios[t] = np.random.uniform(2.5, 12.0)
                ev_ebitda_ratios[t] = np.random.uniform(8.0, 22.0)
                ebit_growths[t] = np.random.uniform(0.05, 0.45)
                ebitda_growths[t] = np.random.uniform(0.05, 0.40)
                ebitda_margins[t] = np.random.uniform(0.12, 0.38)
                dcf_potentials[t] = np.random.uniform(0.10, 0.45)

        return {
            "pe_ratios": pe_ratios,
            "pb_ratios": pb_ratios,
            "ev_ebitda_ratios": ev_ebitda_ratios,
            "ebit_growths": ebit_growths,
            "ebitda_growths": ebitda_growths,
            "ebitda_margins": ebitda_margins,
            "dcf_potentials": dcf_potentials,
        }


# ==============================================================================
# 5. ADVANCED RISK & COVARIANCE ESTIMATION ENGINE
# ==============================================================================


class AdvancedRiskEngine:
    """GARCH/EWMA, Historical Covariance ve Shrinkage teknikleri sunan risk motoru."""

    def __init__(
        self, returns_df, ewma_span=60, garch_weight=0.50, lambda_ridge=1e-4
    ):
        self.returns_df = returns_df
        self.tickers = returns_df.columns
        self.num_assets = len(self.tickers)
        self.ewma_span = ewma_span
        self.garch_weight = garch_weight
        self.lambda_ridge = lambda_ridge

    def calculate_historical_cov(self):
        return self.returns_df.cov() * GLOBAL_CONFIG["TRADING_DAYS_YEAR"]

    def calculate_ewma_cov(self):
        ewma = (
            self.returns_df.ewm(span=self.ewma_span)
            .cov()
            .iloc[-self.num_assets :]
            * GLOBAL_CONFIG["TRADING_DAYS_YEAR"]
        )
        return pd.DataFrame(
            ewma.values, index=self.tickers, columns=self.tickers
        )

    def build_hybrid_covariance_matrix(self, mode="Hibrit (EWMA + Sample Cov)"):
        if mode == "Sample Covariance":
            cov_val = self.calculate_historical_cov().values
        elif mode == "EWMA Covariance":
            cov_val = self.calculate_ewma_cov().values
        else:
            sample_cov = self.calculate_historical_cov().values
            ewma_cov = self.calculate_ewma_cov().values
            cov_val = (self.garch_weight * ewma_cov) + (
                (1.0 - self.garch_weight) * sample_cov
            )

        robust_cov = cov_val + np.eye(self.num_assets) * self.lambda_ridge
        return pd.DataFrame(
            robust_cov, index=self.tickers, columns=self.tickers
        )


# ==============================================================================
# 6. STOCHASTIC MACRO ENGINE (ORNSTEIN-UHLENBECK INTEREST RATE PROCESS)
# ==============================================================================


class StochasticMacroEngine:
    """Risk-Free Rate (Rf) için Vasicek / Ornstein-Uhlenbeck stochastic differential equation simülatörü."""

    @staticmethod
    def run_ornstein_uhlenbeck(
        initial_rate=0.45,
        target_rate=0.35,
        kappa=0.20,
        sigma=0.04,
        days=252,
        paths=100,
        seed=42,
    ):
        np.random.seed(seed)
        dt = 1.0 / days
        rates = np.zeros((days, paths))
        rates[0] = initial_rate

        for t in range(1, days):
            dw = np.random.normal(0, np.sqrt(dt), paths)
            rates[t] = (
                rates[t - 1]
                + kappa * (target_rate - rates[t - 1]) * dt
                + sigma * dw
            )

        return rates

    @staticmethod
    def extract_effective_rf(simulated_paths):
        return float(np.mean(simulated_paths[-1, :]))


# ==============================================================================
# 7. FUNDAMENTAL CATALYST & ALPHA GENERATION ENGINE
# ==============================================================================


class FundamentalCatalystEngine:
    """Çarpanlar, Büyüme ve Marj Verilerini Analiz Eden Alfa Motoru."""

    def __init__(self, fundamental_data):
        self.pe_ratios = fundamental_data.get("pe_ratios", {})
        self.pb_ratios = fundamental_data.get("pb_ratios", {})
        self.ev_ebitda_ratios = fundamental_data.get("ev_ebitda_ratios", {})
        self.ebit_growths = fundamental_data.get("ebit_growths", {})
        self.ebitda_growths = fundamental_data.get("ebitda_growths", {})
        self.ebitda_margins = fundamental_data.get("ebitda_margins", {})

    def calculate_valuation_score(self, ticker):
        pe = self.pe_ratios.get(ticker, 12.0)
        pb = self.pb_ratios.get(ticker, 2.0)
        ev_ebitda = self.ev_ebitda_ratios.get(ticker, 8.0)

        if pe <= 0:
            pe_score = 0.5
        else:
            pe_score = max(0.0, (25.0 - pe) / 25.0)

        if pb <= 0:
            pb_score = 0.5
        else:
            pb_score = max(0.0, (6.0 - pb) / 6.0)

        if ev_ebitda <= 0:
            ev_score = 0.5
        else:
            ev_score = max(0.0, (18.0 - ev_ebitda) / 18.0)

        return (pe_score * 0.35) + (pb_score * 0.30) + (ev_score * 0.35)

    def calculate_growth_score(self, ticker):
        ebit_growth = self.ebit_growths.get(ticker, 0.15)
        ebitda_growth = self.ebitda_growths.get(ticker, 0.15)

        ebit_score = np.clip(ebit_growth, -0.3, 0.7)
        ebitda_score = np.clip(ebitda_growth, -0.3, 0.7)

        return (ebit_score * 0.50) + (ebitda_score * 0.50)

    def generate_all_catalyst_premiums(self):
        premiums = {}
        for ticker in self.pe_ratios.keys():
            v_score = self.calculate_valuation_score(ticker)
            g_score = self.calculate_growth_score(ticker)
            total_catalyst = (v_score * 0.50) + (g_score * 0.50)
            premiums[ticker] = np.clip(total_catalyst * 0.15, -0.10, 0.18)
        return premiums


# ==============================================================================
# 8. MASTER QUANTAMENTAL OPTIMIZER (CORE ENGINE)
# ==============================================================================


def calculate_master_integrated_opt(
    returns_df,
    base_rf=BASE_RF_RATE,
    fundamentals=None,
    min_asset_floor=MIN_ASSET_WEIGHT,
    max_asset_cap=MAX_ASSET_WEIGHT,
    max_bist_cap=MAX_BIST_CAP,
    cov_mode="Hibrit (EWMA + Sample Cov)",
):
    raw_mean_returns = (
        returns_df.mean() * GLOBAL_CONFIG["TRADING_DAYS_YEAR"]
    )
    num_assets = len(raw_mean_returns)
    tickers = returns_df.columns

    if fundamentals is None:
        ingestor_tmp = FinancialDataIngestor(tickers)
        fundamentals = ingestor_tmp.fetch_fundamental_metrics()

    ou_sim_paths = StochasticMacroEngine.run_ornstein_uhlenbeck(
        initial_rate=base_rf, target_rate=base_rf * 0.85
    )
    effective_rf = StochasticMacroEngine.extract_effective_rf(ou_sim_paths)

    cat_engine = FundamentalCatalystEngine(fundamentals)
    catalyst_premiums = cat_engine.generate_all_catalyst_premiums()

    blended_returns = {}
    for t in tickers:
        quant_ret = raw_mean_returns[t]
        dcf_pot = fundamentals["dcf_potentials"].get(t, quant_ret)
        cat_prem = catalyst_premiums.get(t, 0.0)

        blended_returns[t] = (
            (quant_ret * 0.50) + (dcf_pot * 0.30) + (cat_prem * 0.20)
        )

    adjusted_returns = pd.Series(blended_returns)

    risk_engine = AdvancedRiskEngine(returns_df)
    hybrid_cov_df = risk_engine.build_hybrid_covariance_matrix(mode=cov_mode)
    cov_values = hybrid_cov_df.values

    # Kısıtlamalar (Constraints)
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

    bounds = tuple(
        (float(min_asset_floor), float(max_asset_cap))
        for _ in range(num_assets)
    )
    initial_weights = np.ones(num_assets) / num_assets

    def master_objective(weights):
        p_ret = np.dot(weights, adjusted_returns.values)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_values, weights)))

        port_daily_ret = returns_df.dot(weights)
        var_95 = np.percentile(port_daily_ret, 5)
        tail_losses = port_daily_ret[port_daily_ret <= var_95]
        cvar_tail = (
            abs(tail_losses.mean()) if len(tail_losses) > 0 else abs(var_95)
        )

        if p_vol == 0 or cvar_tail == 0:
            return 1e6

        sharpe = (p_ret - effective_rf) / p_vol
        cvar_penalty = cvar_tail * np.sqrt(GLOBAL_CONFIG["TRADING_DAYS_YEAR"])
        hhi_penalty = np.sum(weights**2)

        return -(sharpe - 0.4 * cvar_penalty - 0.05 * hhi_penalty)

    opt_res = minimize(
        master_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": GLOBAL_CONFIG["MAX_OPTIMIZATION_ITERATIONS"],
            "ftol": GLOBAL_CONFIG["TOLERANCE"],
        },
    )

    opt_w = np.maximum(opt_res.x, 0.0)
    opt_w = opt_w / np.sum(opt_w)

    return (
        pd.Series(opt_w, index=returns_df.columns),
        hybrid_cov_df,
        effective_rf,
        adjusted_returns,
        ou_sim_paths,
    )


# ==============================================================================
# 9. PORTFOLIO ANALYTICS & RISK METRICS
# ==============================================================================


class PortfolioAnalytics:

    def __init__(self, weights, returns_df, cov_df, rf):
        self.weights = weights
        self.returns_df = returns_df
        self.cov_df = cov_df
        self.rf = rf
        self.daily_portfolio_returns = returns_df.dot(weights)

    def calculate_annualized_return(self, adjusted_returns_series):
        return np.dot(self.weights, adjusted_returns_series)

    def calculate_annualized_volatility(self):
        return np.sqrt(
            np.dot(self.weights.T, np.dot(self.cov_df.values, self.weights))
        )

    def calculate_sharpe_ratio(self, p_ret, p_vol):
        return (p_ret - self.rf) / p_vol if p_vol > 0 else 0.0

    def calculate_var_cvar(self, alpha=0.95):
        cutoff_percentile = (1 - alpha) * 100
        var_daily = np.percentile(
            self.daily_portfolio_returns, cutoff_percentile
        )
        tail_returns = self.daily_portfolio_returns[
            self.daily_portfolio_returns <= var_daily
        ]
        cvar_daily = tail_returns.mean()

        ann_factor = np.sqrt(GLOBAL_CONFIG["TRADING_DAYS_YEAR"])
        return abs(var_daily) * ann_factor, abs(cvar_daily) * ann_factor

    def calculate_max_drawdown(self):
        cum_returns = (1 + self.daily_portfolio_returns).cumprod()
        peak = cum_returns.cummax()
        drawdown = (cum_returns - peak) / peak
        return abs(drawdown.min())

    def calculate_diversification_index(self):
        return np.sum(self.weights**2)


# ==============================================================================
# 10. VISUALIZATION ENGINE (SEABORN-FREE)
# ==============================================================================


class QuantTerminalVisualizer:

    @staticmethod
    def plot_portfolio_dashboard(
        weights,
        cov_df,
        adjusted_returns,
        returns_df,
        ou_paths,
        analytics_summary,
    ):
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        # Panel 1: Portföy Dağılımı
        ax1 = axes[0, 0]
        colors = [
            "#1f77b4" if t.endswith(".IS") else "#ff7f0e"
            for t in weights.index
        ]
        weights.plot(kind="bar", ax=ax1, color=colors, edgecolor="black")
        ax1.set_title(
            "Optimal Portföy Ağırlıkları (Mavi: BIST, Turuncu: Global/ETF)",
            fontsize=11,
            fontweight="bold",
        )
        ax1.set_ylabel("Ağırlık Oranı")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Panel 2: Risk vs. Getiri
        ax2 = axes[0, 1]
        vols = np.sqrt(np.diag(cov_df.values))
        ax2.scatter(
            vols * 100,
            adjusted_returns.values * 100,
            color="red",
            s=80,
            alpha=0.7,
            label="Bireysel Varlıklar",
        )

        for i, txt in enumerate(weights.index):
            ax2.annotate(
                txt,
                (vols[i] * 100 + 0.3, adjusted_returns[txt] * 100),
                fontsize=8,
            )

        p_vol = analytics_summary["volatility"] * 100
        p_ret = analytics_summary["return"] * 100
        ax2.scatter(
            [p_vol],
            [p_ret],
            color="green",
            s=220,
            marker="*",
            label="OPTIMAL PORTFÖY",
        )
        ax2.set_title(
            "Risk vs. Getiri Uzayı (Volatilite vs Beklenen Getiri)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.set_xlabel("Yıllık Volatilite (%)")
        ax2.set_ylabel("Düzeltilmiş Beklenen Getiri (%)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        # Panel 3: Monte Carlo / OU Faiz Simülasyonu
        ax3 = axes[1, 0]
        ax3.plot(ou_paths[:, :30], alpha=0.25, color="gray")
        ax3.plot(
            ou_paths.mean(axis=1),
            color="blue",
            linewidth=2.5,
            label="Ortalama Beklenen Faiz (Rf)",
        )
        ax3.set_title(
            "Ornstein-Uhlenbeck Makro Faiz (Rf) Simülasyonu",
            fontsize=11,
            fontweight="bold",
        )
        ax3.set_xlabel("İş Günü")
        ax3.set_ylabel("Faiz Oranı")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.5)

        # Panel 4: Tarihsel Kumulatif Getiri
        ax4 = axes[1, 1]
        port_daily = returns_df.dot(weights)
        cum_rets = (1 + port_daily).cumprod() - 1
        ax4.plot(
            cum_rets.index,
            cum_rets.values * 100,
            color="purple",
            linewidth=2,
            label="Quantamental Portföy",
        )
        ax4.set_title(
            "Tarihsel Backtest Kumulatif Getiri (%)",
            fontsize=11,
            fontweight="bold",
        )
        ax4.set_ylabel("Kumulatif Getiri (%)")
        ax4.legend()
        ax4.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)


# ==============================================================================
# 11. MAIN EXECUTION PIPELINE FOR STREAMLIT
# ==============================================================================


def main():
    st.markdown(
        '<div class="main-title">🚀 Quantamental Portföy Optimizasyon Terminali</div>',
        unsafe_allow_allowed_html=True,
    )
    st.markdown(
        '<div class="sub-title">Çok varlıklı (Hisse, ETF, Emtia, Gayrimenkul) Cantillon & Quant-based Portföy Mimarisi</div>',
        unsafe_allow_allowed_html=True,
    )

    if len(TARGET_TICKERS) < 2:
        st.warning(
            "Lütfen optimizasyon yapabilmek için sol panelden en az 2 geçerli varlık giriniz."
        )
        st.stop()

    with st.spinner("Finansal veriler işleniyor ve optimizasyon motoru çalıştırılıyor..."):
        ingestor = FinancialDataIngestor(TARGET_TICKERS)

        if DATA_SOURCE_CHOICE == "Yahoo Finance (Canlı/Tarihsel)":
            returns_dataframe = ingestor.fetch_real_data()
        else:
            returns_dataframe = ingestor.generate_synthetic_returns()

        fundamentals = ingestor.fetch_fundamental_metrics()

        if returns_dataframe.empty or returns_dataframe.shape[1] < 2:
            st.error(
                "Girdiğiniz varlıklar için yeterli veri çekilemedi. Ticker kodlarını kontrol ediniz."
            )
            st.stop()

        (
            opt_weights,
            hybrid_cov,
            effective_rf,
            adjusted_returns,
            ou_paths,
        ) = calculate_master_integrated_opt(
            returns_df=returns_dataframe,
            base_rf=BASE_RF_RATE,
            fundamentals=fundamentals,
            min_asset_floor=MIN_ASSET_WEIGHT,
            max_asset_cap=MAX_ASSET_WEIGHT,
            max_bist_cap=MAX_BIST_CAP,
            cov_mode=ESTIMATION_METHOD,
        )

        analytics = PortfolioAnalytics(
            opt_weights, returns_dataframe, hybrid_cov, effective_rf
        )

        p_ret = analytics.calculate_annualized_return(adjusted_returns)
        p_vol = analytics.calculate_annualized_volatility()
        sharpe = analytics.calculate_sharpe_ratio(p_ret, p_vol)
        var_95, cvar_95 = analytics.calculate_var_cvar(
            alpha=GLOBAL_CONFIG["CONFIDENCE_LEVEL_95"]
        )
        max_dd = analytics.calculate_max_drawdown()
        hhi = analytics.calculate_diversification_index()

    # Özet Tablo Gösterimi
    st.subheader("📊 Optimal Portföy Varlık Dağılımı ve Bilanço Metrikleri")

    df_summary = pd.DataFrame(
        {
            "Ağırlık (%)": (opt_weights * 100).round(2),
            "Beklenen Getiri (%)": (adjusted_returns * 100).round(2),
            "Bireysel Volatilite (%)": (
                np.sqrt(np.diag(hybrid_cov.values)) * 100
            ).round(2),
            "F/K Oranı": pd.Series(fundamentals["pe_ratios"]).round(2),
            "PD/DD Oranı": pd.Series(fundamentals["pb_ratios"]).round(2),
            "EBITDA Marjı (%)": (
                pd.Series(fundamentals["ebitda_margins"]) * 100
            ).round(2),
        }
    )

    st.dataframe(df_summary, use_container_width=True)

    # KPI Metrik Kartları
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    col1.metric("Beklenen Getiri", f"%{p_ret * 100:.2f}")
    col2.metric("Portföy Volatilitesi", f"%{p_vol * 100:.2f}")
    col3.metric("Entegre Sharpe", f"{sharpe:.3f}")
    col4.metric("Yıllık VaR (%95)", f"%{var_95 * 100:.2f}")
    col5.metric("Yıllık CVaR (%95)", f"%{cvar_95 * 100:.2f}")
    col6.metric("Max Drawdown", f"%{max_dd * 100:.2f}")

    # Görselleştirme
    st.markdown("---")
    st.subheader("📈 Performans ve Risk Analiz Paneli")

    analytics_summary = {"return": p_ret, "volatility": p_vol}

    QuantTerminalVisualizer.plot_portfolio_dashboard(
        weights=opt_weights,
        cov_df=hybrid_cov,
        adjusted_returns=adjusted_returns,
        returns_df=returns_dataframe,
        ou_paths=ou_paths,
        analytics_summary=analytics_summary,
    )

    # Ek Analiz: Matplotlib Tabanlı Korelasyon Isı Haritası (Seaborn gerektirmez)
    with st.expander("🔍 Varlık Korelasyon Matrisini İncele"):
        corr_matrix = returns_dataframe.corr().values
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        cax = ax_corr.matshow(corr_matrix, cmap="coolwarm")
        fig_corr.colorbar(cax)

        ticks = np.arange(0, len(returns_dataframe.columns), 1)
        ax_corr.set_xticks(ticks)
        ax_corr.set_yticks(ticks)
        ax_corr.set_xticklabels(returns_dataframe.columns, rotation=45, ha="left")
        ax_corr.set_yticklabels(returns_dataframe.columns)

        for i in range(len(returns_dataframe.columns)):
            for j in range(len(returns_dataframe.columns)):
                ax_corr.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                )

        st.pyplot(fig_corr)


if __name__ == "__main__":
    main()
