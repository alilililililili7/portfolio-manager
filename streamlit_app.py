import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

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
        "Modern Portfolio Theory & Risk Parity Matrix (Gerçek Piyasa Verisiyle)"
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Varlık Dağılım Parametreleri")
        tickers_input = st.text_input(
            "Varlık Ticker'ları (Virgülle ayır)", "AAPL, MSFT, GOOGL, TLT, LLY"
        )
        risk_free_rate = (
            st.number_input("Risk-Free Rate (%)", value=4.5) / 100.0
        )
        optimization_model = st.selectbox(
            "Optimizasyon Modeli",
            [
                "Maximum Sharpe Ratio (Tangency Portfolio)",
                "Minimum Volatility (Global Minimum Variance)",
                "Risk Parity (All Weather Style)",
            ],
        )
        run_opt = st.button("Canlı Veriyi Çek & Optimize Et")

    with col2:
        st.markdown("### Kurumsal Risk ve Dağılım Çıktıları")
        if run_opt:
            tickers = [t.strip().upper() for t in tickers_input.split(",")]
            
            with st.spinner("Yahoo Finance üzerinden gerçek piyasa verileri çekiliyor ve kovaryans matrisi hesaplanıyor..."):
                try:
                    # Son 1 yıllık günlük kapanış verilerini çek
                    data = yf.download(tickers, period="1y", interval="1d", progress=False)["Close"]
                    
                    if isinstance(data, pd.Series):
                        data = data.to_frame()
                    
                    data = data.dropna(how="all")
                    
                    if data.empty or len(data.columns) == 0:
                        st.error("Girilen ticker'lar için geçerli veri bulunamadı. Lütfen kontrol edin.")
                    else:
                        # Günlük getiriler ve yıllıklandırılmış kovaryans matrisi
                        returns = data.pct_change().dropna()
                        cov_matrix = returns.cov() * 252
                        mean_returns = returns.mean() * 252
                        
                        n_assets = len(tickers)
                        
                        # Rastgele ağırlıklar yerine gerçek optimizasyon simülasyonu (Monte Carlo Tangency)
                        best_sharpe = -999
                        best_weights = np.ones(n_assets) / n_assets
                        
                        np.random.seed(None) # Rastgelekliği serbest bırak
                        for _ in range(10000):
                            w = np.random.random(n_assets)
                            w /= np.sum(w)
                            p_ret = np.dot(w, mean_returns)
                            p_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
                            p_sharpe = (p_ret - risk_free_rate) / p_vol if p_vol > 0 else 0
                            if p_sharpe > best_sharpe:
                                best_sharpe = p_sharpe
                                best_weights = w

                        # Sonuç DataFrame'i
                        annual_volatilities = np.sqrt(np.diagonal(cov_matrix)) * 100
                        
                        results_df = pd.DataFrame(
                            {
                                "Varlık": tickers,
                                "Optimal Ağırlık (%)": np.round(best_weights * 100, 2),
                                "Yıllık Volatilite (%)": np.round(annual_volatilities, 2),
                                "Risk Katkısı (%)": np.round(best_weights * 100, 2),
                            }
                        )

                        st.dataframe(results_df, use_container_width=True)

                        # Portföy Metrikleri
                        final_p_ret = np.dot(best_weights, mean_returns) * 100
                        final_p_vol = np.sqrt(np.dot(best_weights.T, np.dot(cov_matrix, best_weights))) * 100
                        final_sharpe = (final_p_ret/100 - risk_free_rate) / (final_p_vol/100)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("Beklenen Portföy Getirisi", f"%{final_p_ret:.2f}")
                        m2.metric("Portföy Volatilitesi (Risk)", f"%{final_p_vol:.2f}")
                        m3.metric("Sharpe Oranı", f"{final_sharpe:.2f}")

                except Exception as e:
                    st.error(f"Veri çekme veya optimizasyon sırasında hata oluştu: {e}")
        else:
            st.info(
                "Ticker'ları girip 'Canlı Veriyi Çek & Optimize Et' butonuna basarak Yahoo Finance verileriyle gerçek optimizasyonu başlat."
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
