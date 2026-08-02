import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Modern Portfolio Theory Engine",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Modern Portfolio Theory & Robust Matrix Optimization Engine")

# ==========================================
# TAB NAVIGATION
# ==========================================
tab1, tab2, tab3 = st.tabs(
    ["Portfolio Optimization (MPT)", "Alternative Assets (PE & VC)", "Settings"]
)

with tab1:
    col_left, col_right = st.columns([1, 2], gap="large")

    # --------------------------------------
    # LEFT COLUMN: INPUT PARAMETERS
    # --------------------------------------
    with col_left:
        st.subheader("Varlık Dağılım Parametreleri")

        # Default tickers from the interface
        ticker_input = st.text_input(
            "Varlık Ticker'ları (Virgülle ayır)",
            value="NVDA,LLY,JPM,AAPL,GOOG,CMND,GCTK,OTLK,INFQ",
        )

        risk_free_input = st.number_input(
            "Risk-Free Rate (%)",
            min_value=0.0,
            max_value=20.0,
            value=3.75,
            step=0.25,
        )

        opt_model = st.selectbox(
            "Optimizasyon Modeli",
            options=["Analitik Kovaryans Tersi (Lagrange Exact Solution)"],
        )

        run_btn = st.button("Matematiksel Optimizasyonu Çalıştır")

    # Parse Tickers
    tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

    if run_btn or len(tickers) > 0:
        # --------------------------------------
        # DATA FETCHING & CLEANING ENGINE
        # --------------------------------------
        try:
            # 5 Yıllık veri çek
            raw_data = yf.download(
                tickers, period="5y", progress=False, auto_adjust=True
            )

            if isinstance(raw_data.columns, pd.MultiIndex):
                if "Close" in raw_data.columns.levels[0]:
                    df_close = raw_data["Close"]
                else:
                    df_close = raw_data.xs("Close", axis=1, level=0)
            else:
                df_close = raw_data[["Close"]] if "Close" in raw_data else raw_data

            df_close = df_close.dropna(how="all")

            # Ortak tarih penceresini bul (Inner Join)
            df_common = df_close.dropna(how="any")
            common_days = len(df_common)

            # Logaritmik Getiriler
            log_returns = np.log(df_common / df_common.shift(1)).dropna()

            # --------------------------------------
            # RIGHT COLUMN: OUTPUTS & WARNINGS
            # --------------------------------------
            with col_right:
                st.subheader("Kurumsal Risk ve Dağılım Çıktıları")

                # Yeni Halka Arz (IPO) / Kısa Tarih Veri Uyarısı
                if common_days < 1000:
                    st.warning(
                        f"⚠️ **Dikkat:** Portföydeki bazı varlıklar yeni halka arz olduğu için "
                        f"optimizasyon ortak olan son **{common_days} işlem günü** üzerinden hesaplanıyor."
                    )

                # Yıllıklandırılmış Getiri ve Kovaryans Matrisi
                mean_returns = log_returns.mean() * 252
                cov_matrix = log_returns.cov() * 252

                n_assets = len(tickers)
                rf_rate = risk_free_input / 100.0

                # --------------------------------------
                # ROBUST MATRIX OPTIMIZATION
                # --------------------------------------
                cov_matrix_vals = cov_matrix.values.copy()

                # L2 Ridge Penalty (Singular/Multicollinear Matris Çökmelerini Önler)
                cov_matrix_vals += np.eye(n_assets) * 1e-6

                # Analitik Kovaryans Tersi (Pseudo-Inverse ile MPT Çözümü)
                cov_inv = np.linalg.pinv(cov_matrix_vals)
                excess_returns = (mean_returns - rf_rate).values

                raw_weights = np.dot(cov_inv, excess_returns)

                # Kısa Pozisyon Yasağı (Long-Only Constraint Threshold: %0.01)
                raw_weights = np.maximum(raw_weights, 0.0001)
                opt_weights = raw_weights / np.sum(raw_weights)

                # --------------------------------------
                # RISK & RETURN METRICS
                # --------------------------------------
                portfolio_return = np.sum(mean_returns * opt_weights)
                portfolio_volatility = np.sqrt(
                    np.dot(opt_weights.T, np.dot(cov_matrix_vals, opt_weights))
                )
                sharpe_ratio = (
                    portfolio_return - rf_rate
                ) / portfolio_volatility

                # Bireysel Yıllık Volatiliteler
                individual_vols = np.sqrt(np.diag(cov_matrix_vals))

                # Marginal Contribution to Risk (MCR) & Risk Katkısı (%)
                mcr = np.dot(cov_matrix_vals, opt_weights) / portfolio_volatility
                risk_contribution_dollars = opt_weights * mcr
                risk_contribution_pct = (
                    risk_contribution_dollars / portfolio_volatility
                ) * 100.0

                # Tablo Oluşturma
                df_results = pd.DataFrame(
                    {
                        "Varlık": tickers,
                        "Optimal Ağırlık (%)": np.round(opt_weights * 100, 2),
                        "Yıllık Volatilite (%)": np.round(
                            individual_vols * 100, 2
                        ),
                        "Risk Katkısı (%)": np.round(risk_contribution_pct, 2),
                    }
                )

                # Tabloyu Göster
                st.dataframe(df_results, use_container_width=True, hide_index=False)

                # Alt Metrik Kartları
                st.write("---")
                m_col1, m_col2, m_col3 = st.columns(3)

                with m_col1:
                    st.caption("Beklenen Portföy Getirisi")
                    st.markdown(f"### %{portfolio_return * 100:.2f}")

                with m_col2:
                    st.caption("Portföy Volatilitesi (Risk)")
                    st.markdown(f"### %{portfolio_volatility * 100:.2f}")

                with m_col3:
                    st.caption("Sharpe Oranı")
                    st.markdown(f"### {sharpe_ratio:.2f}")

        except Exception as e:
            st.error(f"Hesaplama sırasında bir hata oluştu: {str(e)}")
