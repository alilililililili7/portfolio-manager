import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

# Sayfa Yapılandırması
st.set_page_config(
    page_title="Global Multi-Asset Portfolio Manager",
    page_icon="🐆",
    layout="wide",
)

# Oturum Durumu Kontrolü (Giriş ve Disclaimer)
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "disclaimer_accepted" not in st.session_state:
    st.session_state.disclaimer_accepted = False

# 1. Giriş Ekranı
if not st.session_state.logged_in:
    st.title("🔐 Login / Register")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")
    if st.button("Sign In"):
        if email and password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.warning("Lütfen e-posta ve şifre girin.")
    st.stop()

# 2. Yasal Uyarı Ekranı (Legal Disclaimer)
if not st.session_state.disclaimer_accepted:
    st.title("⚠️ Legal Disclaimer & Terms of Use")
    st.info(
        "All portfolio optimizations, risk analyses, PE/VC simulations, and mathematical models presented on this platform are strictly for **educational and informational purposes only**.\n\n"
        "They do **not** constitute professional **Investment Advice**. The developer and platform cannot be held liable for any financial losses incurred based on these simulations."
    )
    accepted = st.checkbox(
        "I have read, understood, and accept that this is not investment advice and all responsibility lies with me."
    )
    if st.button("Agree & Enter App"):
        if accepted:
            st.session_state.disclaimer_accepted = True
            st.rerun()
        else:
            st.error("Devam etmek için şartları kabul etmelisin.")
    st.stop()

# 3. Ana Uygulama Ekranı
st.title("🐆 Global Multi-Asset Portfolio Manager")
st.caption(
    "Institutional Grade Asset Allocation & Risk Parity Engine (Citadel / Bridgewater Standard)"
)

tabs = st.tabs(
    [
        "Portfolio Optimization (MPT)",
        "Alternative Assets (PE & VC)",
        "Settings",
    ]
)

with tabs[0]:
    st.subheader(
        "Modern Portfolio Theory & Matrix Inverse Optimization Engine"
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Varlık Dağılım Parametreleri")
        tickers_input = st.text_input(
            "Varlık Ticker'ları (Virgülle ayır)", "LLY, AVGO, BRK-B, JPM, WMT, MU, AMD"
        )
        risk_free_rate = (
            st.number_input("Risk-Free Rate (%)", value=3.75) / 100.0
        )
        optimization_model = st.selectbox(
            "Optimizasyon Modeli",
            [
                "Analitik Kovaryans Tersi (Lagrange Exact Solution)",
                "Maximum Sharpe Ratio (Bounded SLSQP)",
                "Minimum Volatility (Global Minimum Variance)",
            ],
        )
        run_opt = st.button("Matematiksel Optimizasyonu Çalıştır")

    with col2:
        st.markdown("### Kurumsal Risk ve Dağılım Çıktıları")
        if run_opt:
            tickers = [t.strip().upper() for t in tickers_input.split(",")]
            
            with st.spinner("Piyasa verileri işleniyor ve matris analizi hesaplanıyor..."):
                try:
                    data = yf.download(tickers, period="1y", interval="1d", progress=False)
                    
                    if isinstance(data.columns, pd.MultiIndex):
                        if "Close" in data.columns.levels[0]:
                            data = data["Close"]
                    elif "Close" in data.columns:
                        data = data[["Close"]]
                        data.columns = tickers
                    
                    if isinstance(data, pd.Series):
                        data = data.to_frame()
                    
                    data = data.dropna(how="all")
                    
                    if data.empty or len(data.columns) < len(tickers):
                        st.warning("⚠️ Yahoo Finance veri akışında kısıtlama algılandı. Sentetik matris devreye alındı.")
                        np.random.seed(42)
                        mean_returns = np.array([0.38, 0.35, 0.155, 0.20, 0.15, 0.28, 0.22][:len(tickers)])
                        vols = np.array([0.225, 0.27, 0.14, 0.185, 0.145, 0.38, 0.42][:len(tickers)])
                        cov_matrix = np.outer(vols, vols) * 0.3
                        np.fill_diagonal(cov_matrix, vols**2)
                    else:
                        returns = data.pct_change().dropna()
                        cov_matrix = returns.cov().values * 252
                        mean_returns = returns.mean().values * 252

                    n_assets = len(tickers)
                    
                    def portfolio_performance(weights):
                        p_ret = np.sum(mean_returns * weights)
                        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                        return p_ret, p_vol

                    if "Analitik Kovaryans Tersi" in optimization_model:
                        # Resimdeki 3. Adım: w* = Σ^(-1) * (R - Rf) / 1^T * Σ^(-1) * (R - Rf)
                        cov_inv = np.linalg.pinv(cov_matrix)
                        excess_returns = mean_returns - risk_free_rate
                        
                        raw_weights = np.dot(cov_inv, excess_returns)
                        
                        # Açığa satışı engellemek için negatif ağırlıkları temizleyip normalize ediyoruz
                        raw_weights = np.maximum(raw_weights, 0.01)
                        opt_weights = raw_weights / np.sum(raw_weights)
                        
                    elif "Maximum Sharpe Ratio" in optimization_model:
                        def neg_sharpe_ratio(weights):
                            p_ret, p_vol = portfolio_performance(weights)
                            if p_vol == 0:
                                return 0
                            return -(p_ret - risk_free_rate) / p_vol

                        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                        bounds = tuple((0.01, 1.0) for _ in range(n_assets))
                        init_guess = n_assets * [1.0 / n_assets]

                        result = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
                        opt_weights = result.x
                    else:
                        def portfolio_volatility(weights):
                            return portfolio_performance(weights)[1]

                        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                        bounds = tuple((0.01, 1.0) for _ in range(n_assets))
                        init_guess = n_assets * [1.0 / n_assets]

                        result = minimize(portfolio_volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
                        opt_weights = result.x
                    
                    portfolio_vol = portfolio_performance(opt_weights)[1]
                    marginal_contrib = np.dot(cov_matrix, opt_weights) / portfolio_vol if portfolio_vol > 0 else np.zeros(n_assets)
                    risk_contrib = opt_weights * marginal_contrib
                    risk_contrib_pct = (risk_contrib / portfolio_vol) * 100 if portfolio_vol > 0 else np.zeros(n_assets)

                    annual_vols = np.sqrt(np.diagonal(cov_matrix)) * 100

                    results_df = pd.DataFrame(
                        {
                            "Varlık": tickers,
                            "Optimal Ağırlık (%)": np.round(opt_weights * 100, 2),
                            "Yıllık Volatilite (%)": np.round(annual_vols, 2),
                            "Risk Katkısı (%)": np.round(risk_contrib_pct, 2),
                        }
                    )

                    st.dataframe(results_df, use_container_width=True)

                    final_ret, final_vol = portfolio_performance(opt_weights)
                    final_sharpe = (final_ret - risk_free_rate) / final_vol if final_vol > 0 else 0

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Beklenen Portföy Getirisi", f"%{final_ret * 100:.2f}")
                    m2.metric("Portföy Volatilitesi (Risk)", f"%{final_vol * 100:.2f}")
                    m3.metric("Sharpe Oranı", f"{final_sharpe:.2f}")

                except Exception as e:
                    st.error(f"Optimizasyon motoru hatası: {e}")
        else:
            st.info(
                "Ticker'ları girip 'Matematiksel Optimizasyonu Çalıştır' butonuna basarak matris motorunu ateşle."
            )

with tabs[1]:
    st.subheader("Venture Capital & Private Equity (Max 10% Rule)")
    st.warning(
        "High Risk & Illiquid Asset Tracking: PE/VC allocation cannot exceed 10% of total portfolio value."
    )

    vc_amount = st.number_input(
        "Alternatif Varlık / Girişim Sermayesi Yatırım Tutarı ($)",
        min_value=0,
        value=50000,
    )
    total_portfolio = st.number_input(
        "Toplam Portföy Büyüklüğü ($)", min_value=1, value=600000
    )

    vc_ratio = (vc_amount / total_portfolio) * 100
    st.metric("Mevcut Likidite Dışı Varlık Oranı", f"%{vc_ratio:.2f}")

    if vc_ratio > 10:
        st.error(
            "⚠️ Kural İhlali: Kurumsal risk yönetimi sınırlarına göre likidite dışı varlıklar toplam portföyün %10'unu aşamaz!"
        )
    else:
        st.success(
            "✅ Likidite ve risk dağılımı kurumsal sınırlar içerisinde."
        )

with tabs[2]:
    st.subheader("Account & Language Settings")
    st.selectbox("Select Language", ["English", "Türkçe"])
