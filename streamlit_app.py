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

if not st.session_state.disclaimer_accepted:
    st.title("⚠️ Legal Disclaimer & Terms of Use")
    st.info("Bu platformdaki optimizasyonlar eğitim amaçlıdır. Yatırım tavsiyesi değildir.")
    accepted = st.checkbox("Şartları kabul ediyorum.")
    if st.button("Agree & Enter App"):
        if accepted:
            st.session_state.disclaimer_accepted = True
            st.rerun()
        else:
            st.error("Devam etmek için şartları kabul etmelisin.")
    st.stop()

# Ana Uygulama
st.title("🐆 Global Multi-Asset Portfolio Manager")

tabs = st.tabs(["Portfolio Optimization (MPT)", "Alternative Assets (PE & VC)", "Settings"])

with tabs[0]:
    st.subheader("Modern Portfolio Theory & Matrix Inverse Optimization Engine (5 Years Window)")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Varlık Dağılım Parametreleri")
        tickers_input = st.text_input(
            "Varlık Ticker'ları (Virgülle ayır)", "LLY, AVGO, BRK.B, JPM, WMT, MU, AMD"
        )
        risk_free_rate = st.number_input("Risk-Free Rate (%)", value=3.75) / 100.0
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
            # Ticker formatlama (BRK-B -> BRK.B dönüşümü)
            tickers = [t.strip().upper().replace("-", ".") for t in tickers_input.split(",")]
            
            with st.spinner("Son 5 yıllık piyasa verileri Yahoo Finance üzerinden çekiliyor..."):
                try:
                    # 5 Yıllık Veri Çekimi
                    raw_data = yf.download(tickers, period="5y", interval="1d")["Close"]
                    
                    # Eksik verileri temizle
                    df_close = raw_data.dropna()
                    
                    if df_close.empty or len(df_close.columns) == 0:
                        st.error("Hisseler için geçerli fiyat verisi çekilemedi. Ticker'ları kontrol et.")
                    else:
                        # Günlük Getiriler
                        returns = df_close.pct_change().dropna()
                        
                        # 5 Yıllık Veri Üzerinden Yıllıklandırılmış Kovaryans Matrisi ve Volatilite
                        cov_matrix = returns.cov() * 252
                        annual_vols = np.sqrt(np.diagonal(cov_matrix))
                        
                        # 5 Yıllık Yıllıklandırılmış Ortalama Getiri
                        mean_returns = returns.mean() * 252
                        
                        n_assets = len(tickers)

                        def portfolio_performance(w):
                            p_ret = np.sum(mean_returns * w)
                            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                            return p_ret, p_vol

                        if "Analitik Kovaryans Tersi" in optimization_model:
                            # Matris Tersi Yöntemi: w* = Σ^(-1) * (R - Rf)
                            cov_inv = np.linalg.pinv(cov_matrix.values)
                            excess_returns = (mean_returns - risk_free_rate).values
                            
                            raw_weights = np.dot(cov_inv, excess_returns)
                            
                            # Negatif (Short) pozisyonları sıfırlayıp re-normalize etme
                            raw_weights = np.maximum(raw_weights, 0)
                            if np.sum(raw_weights) > 0:
                                opt_weights = raw_weights / np.sum(raw_weights)
                            else:
                                opt_weights = np.ones(n_assets) / n_assets

                        elif "Maximum Sharpe Ratio" in optimization_model:
                            def neg_sharpe(w):
                                r, v = portfolio_performance(w)
                                return -(r - risk_free_rate) / v if v > 0 else 0

                            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                            bnds = tuple((0.0, 1.0) for _ in range(n_assets))
                            init_w = n_assets * [1.0 / n_assets]
                            res = minimize(neg_sharpe, init_w, method='SLSQP', bounds=bnds, constraints=cons)
                            opt_weights = res.x

                        else: # Minimum Volatility
                            def port_vol(w):
                                return portfolio_performance(w)[1]

                            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
                            bnds = tuple((0.0, 1.0) for _ in range(n_assets))
                            init_w = n_assets * [1.0 / n_assets]
                            res = minimize(port_vol, init_w, method='SLSQP', bounds=bnds, constraints=cons)
                            opt_weights = res.x

                        # Portföy Metrikleri
                        final_ret, final_vol = portfolio_performance(opt_weights)
                        final_sharpe = (final_ret - risk_free_rate) / final_vol if final_vol > 0 else 0

                        # Risk Katkısı (Marginal Contribution to Risk - MCR)
                        port_variance = np.dot(opt_weights.T, np.dot(cov_matrix.values, opt_weights))
                        marginal_contrib = np.dot(cov_matrix.values, opt_weights) / np.sqrt(port_variance)
                        percentage_risk_contrib = (opt_weights * marginal_contrib) / np.sqrt(port_variance) * 100

                        results_df = pd.DataFrame(
                            {
                                "Varlık": tickers,
                                "Optimal Ağırlık (%)": np.round(opt_weights * 100, 2),
                                "Yıllık Volatilite (%)": np.round(annual_vols * 100, 2),
                                "Risk Katkısı (%)": np.round(percentage_risk_contrib, 2),
                            }
                        )

                        st.dataframe(results_df, use_container_width=True)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Beklenen Portföy Getirisi (5Y)", f"%{final_ret * 100:.2f}")
                        m2.metric("Portföy Volatilitesi (Risk)", f"%{final_vol * 100:.2f}")
                        m3.metric("Sharpe Oranı", f"{final_sharpe:.2f}")

                except Exception as e:
                    st.error(f"Optimizasyon hesaplama hatası: {e}")
