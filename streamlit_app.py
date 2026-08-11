import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
import seaborn as sns
import streamlit as st

# Yahoo Finance entegrasyonu (Gerçek veri çekimi için)
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. KONFİGÜRASYON VE GLOSAR (GLOBAL SETTINGS)
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
    "RISK_FREE_RATE": 0.45,  # TR Politika/Gösterge Faiz Oranı (%45)
"TRADING_DAYS_YEAR": 252,
    "CONFIDENCE_LEVEL": 0.95,
    "CONFIDENCE_LEVEL_95": 0.95,
    "CONFIDENCE_LEVEL_99": 0.99,
"DEFAULT_LAMBDA_RIDGE": 1e-4,
    "DEFAULT_MIN_WEIGHT": 0.015,  # Her varlık için minimum %1.5 taban
    "DEFAULT_MAX_WEIGHT": 0.25,  # Her varlık için maksimum %25 tavan
    "DEFAULT_MAX_BIST_CAP": 0.50,  # Portföydeki maks BIST payı %50
    "MAX_OPTIMIZATION_ITERATIONS": 1500,
    "TOLERANCE": 1e-9,
}

# ==============================================================================
# 2. GELİŞMİŞ SENTETİK & REEL DATA FETCH / INGESTION ENGINE
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
    """BIST ve Global hisseler için zaman serisi ve bilanço kalemlerini üreten/çeken modül."""
    """Hisseler, ETF'ler ve Emtialar için veri çekme ve sentetik veri üretme modülü."""

    def __init__(self, tickers, start_date="2024-01-01", end_date="2026-08-11"):
    def __init__(self, tickers, start_date="2023-01-01", end_date="2026-08-11"):
self.tickers = tickers
self.start_date = start_date
self.end_date = end_date

    def fetch_historical_returns(self, seed=42):
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

        drifts = np.random.uniform(0.0003, 0.0012, size=num_assets)
        vols = np.random.uniform(0.012, 0.028, size=num_assets)
        drifts = np.random.uniform(0.0002, 0.0010, size=num_assets)
        vols = np.random.uniform(0.010, 0.025, size=num_assets)

        raw_cov = np.random.uniform(0.15, 0.55, size=(num_assets, num_assets))
        raw_cov = np.random.uniform(0.10, 0.50, size=(num_assets, num_assets))
cov_matrix = (raw_cov + raw_cov.T) / 2
np.fill_diagonal(cov_matrix, 1.0)

@@ -77,27 +241,32 @@ def fetch_fundamental_metrics(self):

for t in self.tickers:
is_bist = t.endswith(".IS")
            pe_ratios[t] = (
                np.random.uniform(4.5, 18.0)
                if is_bist
                else np.random.uniform(14.0, 40.0)
            )
            pb_ratios[t] = (
                np.random.uniform(0.8, 3.8)
                if is_bist
                else np.random.uniform(2.5, 12.0)
            )
            ev_ebitda_ratios[t] = (
                np.random.uniform(3.5, 12.0)
                if is_bist
                else np.random.uniform(8.0, 22.0)
            )

            ebit_growths[t] = np.random.uniform(-0.15, 0.75)
            ebitda_growths[t] = np.random.uniform(-0.10, 0.65)
            ebitda_margins[t] = np.random.uniform(0.08, 0.42)

            dcf_potentials[t] = np.random.uniform(0.10, 0.80)
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
@@ -111,7 +280,7 @@ def fetch_fundamental_metrics(self):


# ==============================================================================
# 3. ADVANCED RISK & COVARIANCE ESTIMATION ENGINE
# 5. ADVANCED RISK & COVARIANCE ESTIMATION ENGINE
# ==============================================================================


@@ -142,28 +311,26 @@ def calculate_ewma_cov(self):
ewma.values, index=self.tickers, columns=self.tickers
)

    def build_hybrid_covariance_matrix(self):
        sample_cov = self.calculate_historical_cov()
        ewma_cov = self.calculate_ewma_cov()

        hybrid_values = (self.garch_weight * ewma_cov.values) + (
            (1.0 - self.garch_weight) * sample_cov.values
        )

        robust_cov = hybrid_values + np.eye(self.num_assets) * self.lambda_ridge
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

    def compute_asset_volatilities(self):
        cov_df = self.build_hybrid_covariance_matrix()
        vols = np.sqrt(np.diag(cov_df.values))
        return pd.Series(vols, index=self.tickers)


# ==============================================================================
# 4. MAKRO SDE & STOCHASTIC INTEREST RATE SIMULATOR (ORNSTEIN-UHLENBECK)
# 6. STOCHASTIC MACRO ENGINE (ORNSTEIN-UHLENBECK INTEREST RATE PROCESS)
# ==============================================================================


@@ -201,12 +368,12 @@ def extract_effective_rf(simulated_paths):


# ==============================================================================
# 5. FUNDAMENTAL CATALYST & ALPHA GENERATION ENGINE
# 7. FUNDAMENTAL CATALYST & ALPHA GENERATION ENGINE
# ==============================================================================


class FundamentalCatalystEngine:
    """Çarpanlar, Büyüme ve Marj Bilanço Verilerini Analiz Eden Alfa Motoru."""
    """Çarpanlar, Büyüme ve Marj Verilerini Analiz Eden Alfa Motoru."""

def __init__(self, fundamental_data):
self.pe_ratios = fundamental_data.get("pe_ratios", {})
@@ -221,136 +388,91 @@ def calculate_valuation_score(self, ticker):
pb = self.pb_ratios.get(ticker, 2.0)
ev_ebitda = self.ev_ebitda_ratios.get(ticker, 8.0)

        pe_score = max(0.0, (20.0 - pe) / 20.0) if pe > 0 else 0.0
        pb_score = max(0.0, (5.0 - pb) / 5.0) if pb > 0 else 0.0
        ev_score = max(0.0, (15.0 - ev_ebitda) / 15.0) if ev_ebitda > 0 else 0.0
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

        ebit_score = np.clip(ebit_growth, -0.4, 0.8)
        ebitda_score = np.clip(ebitda_growth, -0.4, 0.8)
        ebit_score = np.clip(ebit_growth, -0.3, 0.7)
        ebitda_score = np.clip(ebitda_growth, -0.3, 0.7)

return (ebit_score * 0.50) + (ebitda_score * 0.50)

    def calculate_margin_score(self, ticker):
        ebitda_margin = self.ebitda_margins.get(ticker, 0.20)
        return np.clip((ebitda_margin - 0.15) / 0.20, -0.5, 0.5)

def generate_all_catalyst_premiums(self):
premiums = {}
for ticker in self.pe_ratios.keys():
v_score = self.calculate_valuation_score(ticker)
g_score = self.calculate_growth_score(ticker)
            m_score = self.calculate_margin_score(ticker)

            total_catalyst = (
                (v_score * 0.35) + (g_score * 0.40) + (m_score * 0.25)
            )

            premiums[ticker] = np.clip(total_catalyst * 0.18, -0.12, 0.20)

            total_catalyst = (v_score * 0.50) + (g_score * 0.50)
            premiums[ticker] = np.clip(total_catalyst * 0.15, -0.10, 0.18)
return premiums


# ==============================================================================
# 6. YARDIMCI DÜZELTME VE CEZALANDIRMA MODÜLLERİ
# ==============================================================================


def apply_ebitda_margin_penalty(
    blended_returns, ebitda_margins, benchmark_margin=0.15
):
    adjusted = {}
    for ticker, ret in blended_returns.items():
        margin = ebitda_margins.get(ticker, benchmark_margin)
        if margin < benchmark_margin:
            penalty_factor = 1.0 - (benchmark_margin - margin)
            adjusted[ticker] = ret * max(penalty_factor, 0.70)
        else:
            boost_factor = 1.0 + (margin - benchmark_margin) * 0.20
            adjusted[ticker] = ret * min(boost_factor, 1.25)
    return adjusted


# ==============================================================================
# 7. INTEGRATED MASTER QUANTAMENTAL OPTIMIZER (CORE ENGINE)
# 8. MASTER QUANTAMENTAL OPTIMIZER (CORE ENGINE)
# ==============================================================================


def calculate_master_integrated_opt(
returns_df,
    base_rf=GLOBAL_CONFIG["RISK_FREE_RATE"],
    ebitda_margins=None,
    dcf_potentials=None,
    pe_ratios=None,
    pb_ratios=None,
    ev_ebitda_ratios=None,
    ebit_growths=None,
    ebitda_growths=None,
    min_asset_floor=GLOBAL_CONFIG["DEFAULT_MIN_WEIGHT"],
    max_asset_cap=GLOBAL_CONFIG["DEFAULT_MAX_WEIGHT"],
    max_bist_cap=GLOBAL_CONFIG["DEFAULT_MAX_BIST_CAP"],
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

    if ebitda_margins is None:
        ebitda_margins = {t: 0.20 for t in tickers}
    if dcf_potentials is None:
        dcf_potentials = raw_mean_returns.to_dict()
    if pe_ratios is None:
        pe_ratios = {t: 12.0 for t in tickers}
    if pb_ratios is None:
        pb_ratios = {t: 2.0 for t in tickers}
    if ev_ebitda_ratios is None:
        ev_ebitda_ratios = {t: 8.0 for t in tickers}
    if ebit_growths is None:
        ebit_growths = {t: 0.15 for t in tickers}
    if ebitda_growths is None:
        ebitda_growths = {t: 0.15 for t in tickers}
    if fundamentals is None:
        ingestor_tmp = FinancialDataIngestor(tickers)
        fundamentals = ingestor_tmp.fetch_fundamental_metrics()

ou_sim_paths = StochasticMacroEngine.run_ornstein_uhlenbeck(
initial_rate=base_rf, target_rate=base_rf * 0.85
)
effective_rf = StochasticMacroEngine.extract_effective_rf(ou_sim_paths)

    fund_data = {
        "pe_ratios": pe_ratios,
        "pb_ratios": pb_ratios,
        "ev_ebitda_ratios": ev_ebitda_ratios,
        "ebit_growths": ebit_growths,
        "ebitda_growths": ebitda_growths,
        "ebitda_margins": ebitda_margins,
    }
    cat_engine = FundamentalCatalystEngine(fund_data)
    cat_engine = FundamentalCatalystEngine(fundamentals)
catalyst_premiums = cat_engine.generate_all_catalyst_premiums()

blended_returns = {}
for t in tickers:
quant_ret = raw_mean_returns[t]
        dcf_pot = dcf_potentials.get(t, quant_ret)
        dcf_pot = fundamentals["dcf_potentials"].get(t, quant_ret)
cat_prem = catalyst_premiums.get(t, 0.0)

blended_returns[t] = (
            (quant_ret * 0.45) + (dcf_pot * 0.30) + (cat_prem)
            (quant_ret * 0.50) + (dcf_pot * 0.30) + (cat_prem * 0.20)
)

    adjusted_returns_dict = apply_ebitda_margin_penalty(
        blended_returns, ebitda_margins
    )
    adjusted_returns = pd.Series(adjusted_returns_dict)
    adjusted_returns = pd.Series(blended_returns)

risk_engine = AdvancedRiskEngine(returns_df)
    hybrid_cov_df = risk_engine.build_hybrid_covariance_matrix()
    hybrid_cov_df = risk_engine.build_hybrid_covariance_matrix(mode=cov_mode)
cov_values = hybrid_cov_df.values

    # Kısıtlamalar (Constraints)
constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

bist_indices = [
@@ -389,15 +511,18 @@ def master_objective(weights):
cvar_penalty = cvar_tail * np.sqrt(GLOBAL_CONFIG["TRADING_DAYS_YEAR"])
hhi_penalty = np.sum(weights**2)

        return -(sharpe - 0.5 * cvar_penalty - 0.1 * hhi_penalty)
        return -(sharpe - 0.4 * cvar_penalty - 0.05 * hhi_penalty)

opt_res = minimize(
master_objective,
initial_weights,
method="SLSQP",
bounds=bounds,
constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
        options={
            "maxiter": GLOBAL_CONFIG["MAX_OPTIMIZATION_ITERATIONS"],
            "ftol": GLOBAL_CONFIG["TOLERANCE"],
        },
)

opt_w = np.maximum(opt_res.x, 0.0)
@@ -413,11 +538,12 @@ def master_objective(weights):


# ==============================================================================
# 8. PORTFÖY DEĞERLENDİRME & RİSK METRİKLERİ HESAPLAMA MODÜLÜ
# 9. PORTFOLIO ANALYTICS & RISK METRICS
# ==============================================================================


class PortfolioAnalytics:

def __init__(self, weights, returns_df, cov_df, rf):
self.weights = weights
self.returns_df = returns_df
@@ -436,7 +562,7 @@ def calculate_annualized_volatility(self):
def calculate_sharpe_ratio(self, p_ret, p_vol):
return (p_ret - self.rf) / p_vol if p_vol > 0 else 0.0

    def calculate_historical_var_cvar(self, alpha=0.95):
    def calculate_var_cvar(self, alpha=0.95):
cutoff_percentile = (1 - alpha) * 100
var_daily = np.percentile(
self.daily_portfolio_returns, cutoff_percentile
@@ -460,12 +586,11 @@ def calculate_diversification_index(self):


# ==============================================================================
# 9. ADVANCED GRAPHICS & VISUALIZATION ENGINE (İSTEDİĞİN DÜZELTİLMİŞ KISIM)
# 10. VISUALIZATION ENGINE
# ==============================================================================


class QuantTerminalVisualizer:
    """Matplotlib tabanlı grafik, dağılım ve simülasyon görselleştirme modülü."""

@staticmethod
def plot_portfolio_dashboard(
@@ -476,26 +601,25 @@ def plot_portfolio_dashboard(
ou_paths,
analytics_summary,
):
        """Çoklu panelli finansal terminal gösterge paneli çizer."""
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
plt.subplots_adjust(hspace=0.35, wspace=0.25)

        # 1. Panel: Optimal Portföy Ağırlıkları Dağılımı
        # Panel 1: Portföy Dağılımı
ax1 = axes[0, 0]
colors = [
"#1f77b4" if t.endswith(".IS") else "#ff7f0e"
for t in weights.index
]
weights.plot(kind="bar", ax=ax1, color=colors, edgecolor="black")
ax1.set_title(
            "Optimal Portföy Ağırlıkları (Mavi: BIST, Turuncu: ABD)",
            fontsize=12,
            "Optimal Portföy Ağırlıkları (Mavi: BIST, Turuncu: Global/ETF)",
            fontsize=11,
fontweight="bold",
)
ax1.set_ylabel("Ağırlık Oranı")
ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Panel: Risk vs Getiri Profil Karşılaştırması
        # Panel 2: Risk vs. Getiri
ax2 = axes[0, 1]
vols = np.sqrt(np.diag(cov_df.values))
ax2.scatter(
@@ -504,10 +628,9 @@ def plot_portfolio_dashboard(
color="red",
s=80,
alpha=0.7,
            label="Bireysel Hisseler",
            label="Bireysel Varlıklar",
)

        # DÜZELTİLEN KISIM: adjusted_returns[txt] kullanılarak KeyError engellendi
for i, txt in enumerate(weights.index):
ax2.annotate(
txt,
@@ -521,23 +644,23 @@ def plot_portfolio_dashboard(
[p_vol],
[p_ret],
color="green",
            s=200,
            s=220,
marker="*",
label="OPTIMAL PORTFÖY",
)
ax2.set_title(
"Risk vs. Getiri Uzayı (Volatilite vs Beklenen Getiri)",
            fontsize=12,
            fontsize=11,
fontweight="bold",
)
ax2.set_xlabel("Yıllık Volatilite (%)")
ax2.set_ylabel("Düzeltilmiş Beklenen Getiri (%)")
ax2.legend()
ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Panel: Monte Carlo Faiz Simülasyonu Patikaları
        # Panel 3: Monte Carlo / OU Faiz Simülasyonu
ax3 = axes[1, 0]
        ax3.plot(ou_paths[:, :30], alpha=0.3, color="gray")
        ax3.plot(ou_paths[:, :30], alpha=0.25, color="gray")
ax3.plot(
ou_paths.mean(axis=1),
color="blue",
@@ -546,15 +669,15 @@ def plot_portfolio_dashboard(
)
ax3.set_title(
"Ornstein-Uhlenbeck Makro Faiz (Rf) Simülasyonu",
            fontsize=12,
            fontsize=11,
fontweight="bold",
)
        ax3.set_xlabel("İş Günü (252 Gün)")
        ax3.set_xlabel("İş Günü")
ax3.set_ylabel("Faiz Oranı")
ax3.legend()
ax3.grid(True, linestyle="--", alpha=0.5)

        # 4. Panel: Kumulatif Portföy Getiri Patikası
        # Panel 4: Tarihsel Kumulatif Getiri
ax4 = axes[1, 1]
port_daily = returns_df.dot(weights)
cum_rets = (1 + port_daily).cumprod() - 1
@@ -567,172 +690,118 @@ def plot_portfolio_dashboard(
)
ax4.set_title(
"Tarihsel Backtest Kumulatif Getiri (%)",
            fontsize=12,
            fontsize=11,
fontweight="bold",
)
ax4.set_ylabel("Kumulatif Getiri (%)")
ax4.legend()
ax4.grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(
            "🔥 QUANTAMENTAL PORTFOLIO OPTIMIZATION TERMINAL 🔥",
            fontsize=16,
            fontweight="bold",
        )

        # Streamlit ve Standart Çalıştırma Desteği
        try:
            import streamlit as st

            st.pyplot(fig)
        except (ImportError, Exception):
            plt.show()
        st.pyplot(fig)


# ==============================================================================
# 10. EXECUTIVE REPORTING & TERMINAL INTERFACE
# 11. MAIN EXECUTION PIPELINE FOR STREAMLIT
# ==============================================================================


class QuantTerminalReporter:
    @staticmethod
    def print_full_executive_report(
        weights,
        cov_df,
        rf,
        adjusted_returns,
        returns_df,
        fund_metrics,
        analytics,
    ):
        p_ret = analytics.calculate_annualized_return(adjusted_returns)
        p_vol = analytics.calculate_annualized_volatility()
        sharpe = analytics.calculate_sharpe_ratio(p_ret, p_vol)
        var_95, cvar_95 = analytics.calculate_historical_var_cvar()
        max_dd = analytics.calculate_max_drawdown()
        hhi = analytics.calculate_diversification_index()

        bist_w = weights[weights.index.str.endswith(".IS")].sum() * 100
        us_w = 100.0 - bist_w
def main():
    st.markdown(
        '<div class="main-title">🚀 Quantamental Portföy Optimizasyon Terminali</div>',
        unsafe_allow_allowed_html=True,
    )
    st.markdown(
        '<div class="sub-title">Çok varlıklı (Hisse, ETF, Emtia, Gayrimenkul) Cantillon & Quant-based Portföy Mimarisi</div>',
        unsafe_allow_allowed_html=True,
    )

        df_summary = pd.DataFrame(
            {
                "Ağırlık (%)": (weights * 100).round(2),
                "Beklenen Getiri (%)": (adjusted_returns * 100).round(2),
                "Bireysel Volatilite (%)": (
                    np.sqrt(np.diag(cov_df.values)) * 100
                ).round(2),
                "F/K Oranı": pd.Series(fund_metrics["pe_ratios"]).round(2),
                "PD/DD Oranı": pd.Series(fund_metrics["pb_ratios"]).round(2),
                "EBITDA Marjı (%)": (
                    pd.Series(fund_metrics["ebitda_margins"]) * 100
                ).round(2),
            }
    if len(TARGET_TICKERS) < 2:
        st.warning(
            "Lütfen optimizasyon yapabilmek için sol panelden en az 2 geçerli varlık giriniz."
)
        st.stop()

        # Streamlit Varsa Arayüze Bas, Yoksa Terminal Konsoluna
        try:
            import streamlit as st
    with st.spinner("Finansal veriler işleniyor ve optimizasyon motoru çalıştırılıyor..."):
        ingestor = FinancialDataIngestor(TARGET_TICKERS)

            st.title("🚀 Quantamental Portföy Optimizasyon Terminali")
            st.dataframe(df_summary)

            col1, col2, col3 = st.columns(3)
            col1.metric("Beklenen Getiri", f"%{p_ret * 100:.2f}")
            col2.metric("Portföy Volatilitesi", f"%{p_vol * 100:.2f}")
            col3.metric("Entegre Sharpe", f"{sharpe:.3f}")
        if DATA_SOURCE_CHOICE == "Yahoo Finance (Canlı/Tarihsel)":
            returns_dataframe = ingestor.fetch_real_data()
        else:
            returns_dataframe = ingestor.generate_synthetic_returns()

            col4, col5, col6 = st.columns(3)
            col4.metric("Yıllık VaR (%95)", f"%{var_95 * 100:.2f}")
            col5.metric("Yıllık CVaR (%95)", f"%{cvar_95 * 100:.2f}")
            col6.metric("Max Drawdown", f"%{max_dd * 100:.2f}")
        fundamentals = ingestor.fetch_fundamental_metrics()

        except (ImportError, Exception):
            print("\n" + "=" * 80)
            print(
                " 🚀 KURUMSAL QUANTAMENTAL PORTFÖY OPTİMİZASYON TERMINALI RAPORU 🚀"
        if returns_dataframe.empty or returns_dataframe.shape[1] < 2:
            st.error(
                "Girdiğiniz varlıklar için yeterli veri çekilemedi. Ticker kodlarını kontrol ediniz."
)
            print("=" * 80)
            print(df_summary.to_string())
            print("-" * 80)
            print(f"📊 PORFÖY GENEL PERFORMANS METRİKLERİ:")
            print(f" 🟢 Yıllıklandırılmış Beklenen Getiri: %{p_ret * 100:.2f}")
            print(f" 🔴 Yıllıklandırılmış Volatilite    : %{p_vol * 100:.2f}")
            print(f" ⚡ Entegre Sharpe Oranı            : {sharpe:.3f}")
            print(f" 🛡️  Yıllık %95 VaR                 : %{var_95 * 100:.2f}")
            print(f" ⚠️  Yıllık %95 CVaR                : %{cvar_95 * 100:.2f}")
            print(f" 📉 Maksimum Düşüş (Max DD)         : %{max_dd * 100:.2f}")
            print(f" 🧩 HHI Çeşitlendirme Skoru         : {hhi:.4f}")
            print("-" * 80)
            print(f"🌍 BÖLGE DAĞILIMI:")
            print(f" 🇹🇷 BIST Toplam Ağırlığı            : %{bist_w:.2f}")
            print(f" 🇺🇸 ABD Toplam Ağırlığı             : %{us_w:.2f}")
            print("=" * 80 + "\n")


# ==============================================================================
# 11. MAIN ENGINE RUNNER (HEM CONSOLE HEM STREAMLIT UYUMLU)
# ==============================================================================
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

def run_pipeline():
    target_tickers = [
        "THYAO.IS",
        "GARAN.IS",
        "ASELS.IS",
        "KCHOL.IS",
        "BIMAS.IS",
        "AAPL",
        "MSFT",
        "NVDA",
        "AMZN",
    ]
        p_ret = analytics.calculate_annualized_return(adjusted_returns)
        p_vol = analytics.calculate_annualized_volatility()
        sharpe = analytics.calculate_sharpe_ratio(p_ret, p_vol)
        var_95, cvar_95 = analytics.calculate_var_cvar(
            alpha=GLOBAL_CONFIG["CONFIDENCE_LEVEL_95"]
        )
        max_dd = analytics.calculate_max_drawdown()
        hhi = analytics.calculate_diversification_index()

    ingestor = FinancialDataIngestor(
        target_tickers, start_date="2024-01-01", end_date="2026-08-11"
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
    returns_dataframe = ingestor.fetch_historical_returns()
    fundamentals = ingestor.fetch_fundamental_metrics()

    (
        opt_weights,
        hybrid_cov,
        effective_rf,
        adjusted_returns,
        ou_paths,
    ) = calculate_master_integrated_opt(
        returns_df=returns_dataframe,
        base_rf=GLOBAL_CONFIG["RISK_FREE_RATE"],
        ebitda_margins=fundamentals["ebitda_margins"],
        dcf_potentials=fundamentals["dcf_potentials"],
        pe_ratios=fundamentals["pe_ratios"],
        pb_ratios=fundamentals["pb_ratios"],
        ev_ebitda_ratios=fundamentals["ev_ebitda_ratios"],
        ebit_growths=fundamentals["ebit_growths"],
        ebitda_growths=fundamentals["ebitda_growths"],
        min_asset_floor=GLOBAL_CONFIG["DEFAULT_MIN_WEIGHT"],
        max_asset_cap=GLOBAL_CONFIG["DEFAULT_MAX_WEIGHT"],
        max_bist_cap=GLOBAL_CONFIG["DEFAULT_MAX_BIST_CAP"],
    )
    st.dataframe(df_summary, use_container_width=True)

    analytics = PortfolioAnalytics(
        opt_weights, returns_dataframe, hybrid_cov, effective_rf
    )
    # KPI Metrik Kartları
    st.markdown("---")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    QuantTerminalReporter.print_full_executive_report(
        weights=opt_weights,
        cov_df=hybrid_cov,
        rf=effective_rf,
        adjusted_returns=adjusted_returns,
        returns_df=returns_dataframe,
        fund_metrics=fundamentals,
        analytics=analytics,
    )
    col1.metric("Beklenen Getiri", f"%{p_ret * 100:.2f}")
    col2.metric("Portföy Volatilitesi", f"%{p_vol * 100:.2f}")
    col3.metric("Entegre Sharpe", f"{sharpe:.3f}")
    col4.metric("Yıllık VaR (%95)", f"%{var_95 * 100:.2f}")
    col5.metric("Yıllık CVaR (%95)", f"%{cvar_95 * 100:.2f}")
    col6.metric("Max Drawdown", f"%{max_dd * 100:.2f}")

    analytics_summary = {
        "return": analytics.calculate_annualized_return(adjusted_returns),
        "volatility": analytics.calculate_annualized_volatility(),
    }
    # Görselleştirme
    st.markdown("---")
    st.subheader("📈 Performans ve Risk Analiz Paneli")

    analytics_summary = {"return": p_ret, "volatility": p_vol}

QuantTerminalVisualizer.plot_portfolio_dashboard(
weights=opt_weights,
@@ -743,6 +812,19 @@ def run_pipeline():
analytics_summary=analytics_summary,
)

    # Ek Analiz: Korelasyon Isı Haritası
    with st.expander("🔍 Varlık Korelasyon Matrisini İncele"):
        fig_corr, ax_corr = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            returns_dataframe.corr(),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax_corr,
            cbar=True,
        )
        st.pyplot(fig_corr)


if __name__ == "__main__":
    run_pipeline()
    main()
