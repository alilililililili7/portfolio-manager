import sys
import warnings
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. KONFİGÜRASYON VE GLOSAR (GLOBAL SETTINGS)
# ==============================================================================

GLOBAL_CONFIG = {
    "RISK_FREE_RATE": 0.45,  # TR Politika/Gösterge Faiz Oranı (%45)
    "TRADING_DAYS_YEAR": 252,
    "CONFIDENCE_LEVEL": 0.95,
    "DEFAULT_LAMBDA_RIDGE": 1e-4,
    "DEFAULT_MIN_WEIGHT": 0.015,  # Her varlık için minimum %1.5 taban
    "DEFAULT_MAX_WEIGHT": 0.25,  # Her varlık için maksimum %25 tavan
    "DEFAULT_MAX_BIST_CAP": 0.50,  # Portföydeki maks BIST payı %50
}

# ==============================================================================
# 2. GELİŞMİŞ SENTETİK & REEL DATA FETCH / INGESTION ENGINE
# ==============================================================================


class FinancialDataIngestor:
    """BIST ve Global hisseler için zaman serisi ve bilanço kalemlerini üreten/çeken modül."""

    def __init__(self, tickers, start_date="2024-01-01", end_date="2026-08-11"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date

    def fetch_historical_returns(self, seed=42):
        np.random.seed(seed)
        dates = pd.date_range(
            start=self.start_date, end=self.end_date, freq="B"
        )
        num_days = len(dates)
        num_assets = len(self.tickers)

        drifts = np.random.uniform(0.0003, 0.0012, size=num_assets)
        vols = np.random.uniform(0.012, 0.028, size=num_assets)

        raw_cov = np.random.uniform(0.15, 0.55, size=(num_assets, num_assets))
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
# 3. ADVANCED RISK & COVARIANCE ESTIMATION ENGINE
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

    def build_hybrid_covariance_matrix(self):
        sample_cov = self.calculate_historical_cov()
        ewma_cov = self.calculate_ewma_cov()

        hybrid_values = (self.garch_weight * ewma_cov.values) + (
            (1.0 - self.garch_weight) * sample_cov.values
        )

        robust_cov = hybrid_values + np.eye(self.num_assets) * self.lambda_ridge

        return pd.DataFrame(
            robust_cov, index=self.tickers, columns=self.tickers
        )

    def compute_asset_volatilities(self):
        cov_df = self.build_hybrid_covariance_matrix()
        vols = np.sqrt(np.diag(cov_df.values))
        return pd.Series(vols, index=self.tickers)


# ==============================================================================
# 4. MAKRO SDE & STOCHASTIC INTEREST RATE SIMULATOR (ORNSTEIN-UHLENBECK)
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
# 5. FUNDAMENTAL CATALYST & ALPHA GENERATION ENGINE
# ==============================================================================


class FundamentalCatalystEngine:
    """Çarpanlar, Büyüme ve Marj Bilanço Verilerini Analiz Eden Alfa Motoru."""

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

        pe_score = max(0.0, (20.0 - pe) / 20.0) if pe > 0 else 0.0
        pb_score = max(0.0, (5.0 - pb) / 5.0) if pb > 0 else 0.0
        ev_score = max(0.0, (15.0 - ev_ebitda) / 15.0) if ev_ebitda > 0 else 0.0

        return (pe_score * 0.35) + (pb_score * 0.30) + (ev_score * 0.35)

    def calculate_growth_score(self, ticker):
        ebit_growth = self.ebit_growths.get(ticker, 0.15)
        ebitda_growth = self.ebitda_growths.get(ticker, 0.15)

        ebit_score = np.clip(ebit_growth, -0.4, 0.8)
        ebitda_score = np.clip(ebitda_growth, -0.4, 0.8)

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
    catalyst_premiums = cat_engine.generate_all_catalyst_premiums()

    blended_returns = {}
    for t in tickers:
        quant_ret = raw_mean_returns[t]
        dcf_pot = dcf_potentials.get(t, quant_ret)
        cat_prem = catalyst_premiums.get(t, 0.0)

        blended_returns[t] = (
            (quant_ret * 0.45) + (dcf_pot * 0.30) + (cat_prem)
        )

    adjusted_returns_dict = apply_ebitda_margin_penalty(
        blended_returns, ebitda_margins
    )
    adjusted_returns = pd.Series(adjusted_returns_dict)

    risk_engine = AdvancedRiskEngine(returns_df)
    hybrid_cov_df = risk_engine.build_hybrid_covariance_matrix()
    cov_values = hybrid_cov_df.values

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

        return -(sharpe - 0.5 * cvar_penalty - 0.1 * hhi_penalty)

    opt_res = minimize(
        master_objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
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
# 8. PORTFÖY DEĞERLENDİRME & RİSK METRİKLERİ HESAPLAMA MODÜLÜ
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

    def calculate_historical_var_cvar(self, alpha=0.95):
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
# 9. ADVANCED GRAPHICS & VISUALIZATION ENGINE (İSTEDİĞİN DÜZELTİLMİŞ KISIM)
# ==============================================================================


class QuantTerminalVisualizer:
    """Matplotlib tabanlı grafik, dağılım ve simülasyon görselleştirme modülü."""

    @staticmethod
    def plot_portfolio_dashboard(
        weights,
        cov_df,
        adjusted_returns,
        returns_df,
        ou_paths,
        analytics_summary,
    ):
        """Çoklu panelli finansal terminal gösterge paneli çizer."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        # 1. Panel: Optimal Portföy Ağırlıkları Dağılımı
        ax1 = axes[0, 0]
        colors = [
            "#1f77b4" if t.endswith(".IS") else "#ff7f0e"
            for t in weights.index
        ]
        weights.plot(kind="bar", ax=ax1, color=colors, edgecolor="black")
        ax1.set_title(
            "Optimal Portföy Ağırlıkları (Mavi: BIST, Turuncu: ABD)",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_ylabel("Ağırlık Oranı")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # 2. Panel: Risk vs Getiri Profil Karşılaştırması
        ax2 = axes[0, 1]
        vols = np.sqrt(np.diag(cov_df.values))
        ax2.scatter(
            vols * 100,
            adjusted_returns.values * 100,
            color="red",
            s=80,
            alpha=0.7,
            label="Bireysel Hisseler",
        )

        # DÜZELTİLEN KISIM: adjusted_returns[txt] kullanılarak KeyError engellendi
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
            s=200,
            marker="*",
            label="OPTIMAL PORTFÖY",
        )
        ax2.set_title(
            "Risk vs. Getiri Uzayı (Volatilite vs Beklenen Getiri)",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xlabel("Yıllık Volatilite (%)")
        ax2.set_ylabel("Düzeltilmiş Beklenen Getiri (%)")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        # 3. Panel: Monte Carlo Faiz Simülasyonu Patikaları
        ax3 = axes[1, 0]
        ax3.plot(ou_paths[:, :30], alpha=0.3, color="gray")
        ax3.plot(
            ou_paths.mean(axis=1),
            color="blue",
            linewidth=2.5,
            label="Ortalama Beklenen Faiz (Rf)",
        )
        ax3.set_title(
            "Ornstein-Uhlenbeck Makro Faiz (Rf) Simülasyonu",
            fontsize=12,
            fontweight="bold",
        )
        ax3.set_xlabel("İş Günü (252 Gün)")
        ax3.set_ylabel("Faiz Oranı")
        ax3.legend()
        ax3.grid(True, linestyle="--", alpha=0.5)

        # 4. Panel: Kumulatif Portföy Getiri Patikası
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
            fontsize=12,
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


# ==============================================================================
# 10. EXECUTIVE REPORTING & TERMINAL INTERFACE
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
        )

        # Streamlit Varsa Arayüze Bas, Yoksa Terminal Konsoluna
        try:
            import streamlit as st

            st.title("🚀 Quantamental Portföy Optimizasyon Terminali")
            st.dataframe(df_summary)

            col1, col2, col3 = st.columns(3)
            col1.metric("Beklenen Getiri", f"%{p_ret * 100:.2f}")
            col2.metric("Portföy Volatilitesi", f"%{p_vol * 100:.2f}")
            col3.metric("Entegre Sharpe", f"{sharpe:.3f}")

            col4, col5, col6 = st.columns(3)
            col4.metric("Yıllık VaR (%95)", f"%{var_95 * 100:.2f}")
            col5.metric("Yıllık CVaR (%95)", f"%{cvar_95 * 100:.2f}")
            col6.metric("Max Drawdown", f"%{max_dd * 100:.2f}")

        except (ImportError, Exception):
            print("\n" + "=" * 80)
            print(
                " 🚀 KURUMSAL QUANTAMENTAL PORTFÖY OPTİMİZASYON TERMINALI RAPORU 🚀"
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

    ingestor = FinancialDataIngestor(
        target_tickers, start_date="2024-01-01", end_date="2026-08-11"
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

    analytics = PortfolioAnalytics(
        opt_weights, returns_dataframe, hybrid_cov, effective_rf
    )

    QuantTerminalReporter.print_full_executive_report(
        weights=opt_weights,
        cov_df=hybrid_cov,
        rf=effective_rf,
        adjusted_returns=adjusted_returns,
        returns_df=returns_dataframe,
        fund_metrics=fundamentals,
        analytics=analytics,
    )

    analytics_summary = {
        "return": analytics.calculate_annualized_return(adjusted_returns),
        "volatility": analytics.calculate_annualized_volatility(),
    }

    QuantTerminalVisualizer.plot_portfolio_dashboard(
        weights=opt_weights,
        cov_df=hybrid_cov,
        adjusted_returns=adjusted_returns,
        returns_df=returns_dataframe,
        ou_paths=ou_paths,
        analytics_summary=analytics_summary,
    )


if __name__ == "__main__":
    run_pipeline()
