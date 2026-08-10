import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Advanced Multi-Asset Matrix Allocator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏛️ Dinamik Varlık Kategori Seçimli & Matris Tabanlı Portföy Motoru")

# ==========================================
# SIDEBAR: BÜTÇE VE KATEGORİ SEÇİMLERİ
# ==========================================
st.sidebar.header("💰 Bütçe ve Para Birimi")
currency = st.sidebar.selectbox("Para Birimi", options=["USD ($)", "TRY (₺)"])
symbol = "$" if "USD" in currency else "₺"

total_budget = st.sidebar.number_input(
    f"Toplam Bütçe ({symbol})",
    min_value=100.0,
    value=10000.0,
    step=500.0,
)

st.sidebar.write("---")
st.sidebar.subheader("🎯 Varlık Kategorisi Seçimleri")
st.sidebar.caption("İstemediğin kategorinin işaretini kaldırabilirsin.")

# Category Checkboxes
include_stocks = st.sidebar.checkbox("📈 Hisse Senetleri", value=True)
include_etfs = st.sidebar.checkbox("🧺 Tematik / Sektörel ETF'ler", value=True)
include_metals = st.sidebar.checkbox("🥇 Emtia & Değerli Madenler", value=True)
include_reit = st.sidebar.checkbox("🏢 Emlak / GYO", value=True)
include_rf = st.sidebar.checkbox("💵 Faiz / Repo / Risk-Free", value=False) # Varsayılan kapalı (istenebilir)

# Collect User Assets
selected_assets = {}

if include_stocks:
    stocks_in = st.sidebar.text_input("Hisseler (Ticker)", value="NVDA, AAPL, JPM")
    for t in stocks_in.split(","):
        if t.strip(): selected_assets[t.strip().upper()] = "Hisse"

if include_etfs:
    etfs_in = st.sidebar.text_input("ETF'ler (Ticker)", value="QQQ, SMH")
    for t in etfs_in.split(","):
        if t.strip(): selected_assets[t.strip().upper()] = "ETF"

if include_metals:
    metals_in = st.sidebar.text_input("Maden/Emtia (Ticker)", value="GLD, SLV")
    for t in metals_in.split(","):
        if t.strip(): selected_assets[t.strip().upper()] = "Emtia/Maden"

if include_reit:
    reit_in = st.sidebar.text_input("Emlak/GYO (Ticker)", value="VNQ")
    for t in reit_in.split(","):
        if t.strip(): selected_assets[t.strip().upper()] = "Emlak/GYO"

rf_rate = 0.0
if include_rf:
    rf_rate_input = st.sidebar.number_input("Yıllık Faiz Oranı (%)", min_value=0.0, value=4.0, step=0.25)
    rf_rate = rf_rate_input / 100.0

st.sidebar.write("---")
risk_aversion = st.sidebar.slider(
    "Risk Tiksinti Katsayısı (A)",
    min_value=1.0, max_value=10.0, value=3.0, step=0.5,
    help="Yüksek değer daha muhafazakar/düşük volatilite odaklı dağılım yapar."
)

run_opt = st.sidebar.button("🚀 Portföyü Hesapla & Optimize Et", type="primary")

# ==========================================
# MAIN ANALYZER & MATHEMATICAL ENGINE
# ==========================================
tickers = list(selected_assets.keys())

if run_opt or len(tickers) > 0:
    if len(tickers) < 2:
        st.error("Matris ve Korelasyon hesaplaması için lütfen en az 2 aktif varlık (Ticker) seçin.")
    else:
        try:
            # 1. Yahoo Finance Data Pull
            with st.spinner("Piyasa verileri çekiliyor ve matrisler hesaplanıyor..."):
                raw_data = yf.download(tickers, period="3y", progress=False, auto_adjust=True)
                
                if isinstance(raw_data.columns, pd.MultiIndex):
                    df_close = raw_data["Close"] if "Close" in raw_data.columns.levels[0] else raw_data.xs("Close", axis=1, level=0)
                else:
                    df_close = raw_data[["Close"]] if "Close" in raw_data else raw_data
                
                valid_cols = [t for t in tickers if t in df_close.columns]
                df_close = df_close[valid_cols].dropna(how="any")

                # 2. Log Return & Matrix Math
                log_returns = np.log(df_close / df_close.shift(1)).dropna()
                active_tickers = list(log_returns.columns)
                N = len(active_tickers)

                # Annually Metrics
                mean_returns = log_returns.mean() * 252 # Yıllık Beklenen Getiri Vektörü (μ)
                cov_matrix = log_returns.cov() * 252     # Kovaryans Matrisi (Σ)
                corr_matrix = log_returns.corr()         # Korelasyon Matrisi (R)

                # Matrix Inversion (Σ^-1)
                cov_vals = cov_matrix.values + np.eye(N) * 1e-6 # Numerical stability
                cov_inv = np.linalg.pinv(cov_vals)              # Ters Kovaryans Matrisi

                # Tangency Portfolio Optimization (Sharpe Maximization)
                excess_returns = mean_returns.values - rf_rate
                raw_weights = np.dot(cov_inv, excess_returns)
                raw_weights = np.maximum(raw_weights, 0.0) # Long-only constraint
                
                if np.sum(raw_weights) > 0:
                    weights = raw_weights / np.sum(raw_weights)
                else:
                    weights = np.ones(N) / N

                # Portfolio Metrics
                port_return = np.sum(mean_returns.values * weights)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_vals, weights)))
                sharpe_ratio = (port_return - rf_rate) / port_vol if port_vol > 0 else 0.0

                # Budget Allocation Calculation
                allocated_amounts = total_budget * weights

            # ==========================================
            # DISPLAY RESULTS & TABULAR DASHBOARD
            # ==========================================
            st.subheader("📌 1. Optimal Bütçe & Varlık Dağılım Tablosu")
            
            res_df = pd.DataFrame({
                "Ticker": active_tickers,
                "Kategori": [selected_assets[t] for t in active_tickers],
                "Yıllık Beklenen Getiri": [f"%{r*100:.2f}" for r in mean_returns.values],
                "Bireysel Volatilite (Risk)": [f"%{v*100:.2f}" for v in np.sqrt(np.diag(cov_vals))],
                "Optimal Ağırlık (%)": np.round(weights * 100, 2),
                f"Yatırılacak Tutar ({symbol})": np.round(allocated_amounts, 2)
            })

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam Bütçe", f"{symbol}{total_budget:,.2f}")
            m2.metric("Beklenen Yıllık Getiri", f"%{port_return*100:.2f}")
            m3.metric("Portföy Volatilitesi (Risk)", f"%{port_vol*100:.2f}")
            m4.metric("Sharpe Oranı", f"{sharpe_ratio:.2f}")

            c_tbl, c_pie = st.columns([1.3, 1])
            with c_tbl:
                st.dataframe(
                    res_df.style.format({f"Yatırılacak Tutar ({symbol})": f"{symbol}{{:,.2f}}", "Optimal Ağırlık (%)": "%{:.2f}"}),
                    use_container_width=True, hide_index=True
                )
            with c_pie:
                fig_p = px.pie(res_df, values="Optimal Ağırlık (%)", names="Ticker", hole=0.35, title="Para Dağılımı")
                fig_p.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_p, use_container_width=True)

            st.write("---")

            # ==========================================
            # MATRİS ANALİZLERİ (KORELASYON & KOVARYANS)
            # ==========================================
            st.subheader("🧮 2. Portföy Matris Analizi & Risk Metrikleri")
            
            tab_corr, tab_cov, tab_inv = st.tabs([
                "🔗 Korelasyon Matrisi (Correlation)", 
                "📐 Kovaryans Matrisi (Covariance)", 
                "🔄 Ters Kovaryans Matrisi (Inverse Matrix)"
            ])

            with tab_corr:
                st.caption("Korelasyon -1 ile +1 arasındadır. Birbirleriyle korelasyonu düşük (maviye yakın) varlıklar riski düşürür.")
                fig_corr = px.imshow(
                    corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="Varlıklar Arası Korelasyon Matrisi"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            with tab_cov:
                st.caption("Yıllıklandırılmış kovaryans matrisi ($\Sigma$). Diagonaller varlıkların kendi varyansıdır.")
                st.dataframe(cov_matrix.style.format("{:.4f}"), use_container_width=True)

            with tab_inv:
                st.caption("Markowitz MPT optimizasyonunda ağırlıkları ($w = \Sigma^{-1} \cdot \mu$) hesaplamada kullanılan Ters Kovaryans Matrisi ($\Sigma^{-1}$).")
                df_inv = pd.DataFrame(cov_inv, index=active_tickers, columns=active_tickers)
                st.dataframe(df_inv.style.format("{:.4f}"), use_container_width=True)

        except Exception as e:
            st.error(f"Veri çekme veya matris hesaplama hatası: {str(e)}")
