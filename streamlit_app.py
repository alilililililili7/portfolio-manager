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

    with col_left:
        st.subheader("Varlık Dağılım Parametreleri")

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

    # Kullanıcı girdisini temizle ve tekrarları önle (Sırayı koru)
    raw_tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
    tickers = list(dict.fromkeys(raw_tickers))

    if (run_btn or len(tickers) > 0) and len(tickers) > 0:
        try:
            # 5 Yıllık veri çek
            raw_data = yf.download(
                tickers, period="5y", progress=False, auto_adjust=True
            )

            # DataFrame Formatlama (MultiIndex veya Tekil)
            if isinstance(raw_data.columns, pd.MultiIndex):
                if "Close" in raw_data.columns.levels[0]:
                    df_close = raw_data["Close"]
                else:
                    df_close = raw_data.xs("Close", axis=1, level=0)
            else:
                df_close = raw_data[["Close"]] if "Close" in raw_data else raw_data

            # yfinance alfabetik getirebilir, kullanıcının girdiği sıraya zorla!
            valid_cols = [t for t in tickers if t in df_close.columns]
            df_close = df_close[valid_cols]

            # Ortak tarih kesişimi (Inner Join)
            df_common = df_close.dropna(how="any")
            common_days = len(df_common)

            # Logaritmik Getiriler
            log_returns = np.log(df_common / df_common.shift(1)).dropna()

            # Garantili Kolon Listesi (Sıralama Eşleşmesi İçin)
            active_tickers = list(log_returns.columns)
            n_assets = len(active_tickers)

            with col_right:
                st.subheader("Kurumsal Risk ve Dağılım Çıktıları")

                # Halka Arz / Kısıtlı Tarih Uyarısı
                if common_days < 1000:
                    st.warning(
                        f"⚠️ **Dikkat:** Portföydeki bazı varlıklar yeni halka arz olduğu için "
                        f"optimizasyon ortak olan son **{common_days} işlem günü** üzerinden hesaplanıyor."
                    )

                # Yıllıklandırılmış Getiri ve Kovaryans Matrisi (Pandas Series/DataFrame)
                mean_returns_series = log_returns.mean() * 252
                cov_matrix_df = log_returns.cov() * 252

                rf_rate = risk_free_input / 100.0

                # Matris Islemleri
                cov_matrix_vals = cov_matrix_df.values.copy()

                # Ridge Penalty (Çoklu doğrusallık ve tekilleşmeyi engeller)
                cov_matrix_vals += np.eye(n_assets) * 1e-6

                cov_inv = np.linalg.pinv(cov_matrix_vals)
                excess_returns = (mean_returns_series.values - rf_rate)

                raw_weights = np.dot(cov_inv, excess_returns)

                # Long-Only Kısıtı (%0 Tabanı)
                raw_weights = np.maximum(raw_weights, 0.0)
                
                # Toplam ağırlık 0 olursa eşit dağıt (Sıfıra bölünme koruması)
                if np.sum(raw_weights) > 0:
                    opt_weights = raw_weights / np.sum(raw_weights)
                else:
                    opt_weights = np.ones(n_assets) / n_assets

                # Portföy Metrikleri
                portfolio_return = np.sum(mean_returns_series.values * opt_weights)
                portfolio_volatility = np.sqrt(
                    np.dot(opt_weights.T, np.dot(cov_matrix_vals, opt_weights))
                )
                sharpe_ratio = (
                    (portfolio_return - rf_rate) / portfolio_volatility
                    if portfolio_volatility > 0
                    else 0.0
                )

                # Bireysel Yıllık Volatiliteler (Doğru Çapraz Eşleşme)
                individual_vols = np.sqrt(np.diag(cov_matrix_vals))

                # Risk Katkısı (%) Hesaplaması
                mcr = np.dot(cov_matrix_vals, opt_weights) / portfolio_volatility
                risk_contribution_pct = (
                    (opt_weights * mcr) / portfolio_volatility
                ) * 100.0

                # TABLO OLUŞTURMA (active_tickers garantisiyle satır kayması tamamen engellendi)
                df_results = pd.DataFrame(
                    {
                        "Varlık": active_tickers,
                        "Optimal Ağırlık (%)": np.round(opt_weights * 100, 2),
                        "Yıllık Volatilite (%)": np.round(
                            individual_vols * 100, 2
                        ),
                        "Risk Katkısı (%)": np.round(risk_contribution_pct, 2),
                    }
                )

                # Tabloyu Göster
                st.dataframe(df_results, use_container_width=True, hide_index=False)

                # Alt Kartlar
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
