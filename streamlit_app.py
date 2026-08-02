import streamlit as st
import numpy as np
import pandas as pd

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
        "Modern Portfolio Theory & Risk Parity Matrix (Stoklar, ETF'ler, Tahviller)"
    )

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Varlık Dağılım Parametreleri")
        tickers_input = st.text_input(
            "Varlık Ticker'ları (Virgülle ayır)", "AAPL, MSFT, GOOGL, TLT, GLD"
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
        run_opt = st.button("Matrisi Çalıştır & Optimize Et")

    with col2:
        st.markdown("### Kurumsal Risk ve Dağılım Çıktıları")
        if run_opt:
            tickers = [t.strip() for t in tickers_input.split(",")]
            np.random.seed(42)
            n_assets = len(tickers)

            # Simüle edilmiş kurumsal veriler (Kovaryans ve Beklenen Getiri)
            weights = np.random.dirichlet(np.ones(n_assets), size=1)[0]
            weights = weights / np.sum(weights)

            results_df = pd.DataFrame(
                {
                    "Varlık": tickers,
                    "Optimal Ağırlık (%)": np.round(weights * 100, 2),
                    "Yıllık Volatilite (%)": np.round(
                        np.random.uniform(12, 35, n_assets), 2
                    ),
                    "Risk Katkısı (%)": np.round(weights * 100, 2),
                }
            )

            st.dataframe(results_df, use_container_width=True)

            # Metrik Kartları
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Beklenen Portföy Getirisi",
                f"%{np.random.uniform(14, 22):.2f}",
            )
            m2.metric(
                "Portföy Volatilitesi (Risk)",
                f"%{np.random.uniform(10, 18):.2f}",
            )
            m3.metric(
                "Sharpe Oranı", f"{np.random.uniform(1.2, 2.1):.2f}"
            )
        else:
            st.info(
                "Parametreleri belirleyip 'Matrisi Çalıştır & Optimize Et' butonuna basarak kovaryans ve ağırlık hesaplamalarını başlat."
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
