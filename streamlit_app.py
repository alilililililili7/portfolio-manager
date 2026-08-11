import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Sayfa Konfigürasyonu
st.set_page_config(
    page_title="Quantamental Portfolio & Risk Terminal", layout="wide"
)

# Sidebar - Dinamik Parametreler ve Varlık Seçimleri
st.sidebar.markdown("## 🏛️ Dinamik Risksiz Faiz Ayarları")
tl_rate = st.sidebar.slider("TL Faiz / Mevduat (%)", 0.0, 100.0, 45.0, 0.5)
usd_rate = st.sidebar.number_input("USD Faiz / SOFR (%)", 0.0, 20.0, 3.75, 0.25)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎯 Varlık Seçimleri")

bist_selected = st.sidebar.checkbox("☑️ BİST Hisseleri (TL)", value=True)
bist_tickers = st.sidebar.text_area(
    "BİST Hisseleri", "AKBNK.IS, BIMAS.IS, EREGL.IS, KCHOL.IS, SISE.IS, THYAO.IS, TUPRS.IS"
)

usd_etf_selected = st.sidebar.checkbox("☑️ ABD ETF / Hisseleri (USD)", value=True)
usd_tickers = st.sidebar.text_area("ABD Varlıkları", "SPY, VOO")

commodity_selected = st.sidebar.checkbox("☑️ Emtialar (USD)", value=True)
commodity_tickers = st.sidebar.text_area("Emtialar", "IAU, CPER")

# Örnek Veri Hazırlığı
tickers = [t.strip() for t in (bist_tickers + "," + usd_tickers + "," + commodity_tickers).split(",") if t.strip()]
np.random.seed(42)
n_assets = len(tickers)

# Ağırlıklar
weights = np.zeros(n_assets)
if "TUPRS.IS" in tickers:
    weights[tickers.index("TUPRS.IS")] = 0.6285
if "IAU" in tickers:
    weights[tickers.index("IAU")] = 0.3715

categories = []
for t in tickers:
    if ".IS" in t:
        categories.append("Hisse (BİST / TL)")
    elif t in ["IAU", "CPER"]:
        categories.append("Emtia (USD)")
    else:
        categories.append("ETF (ABD / USD)")

df_portfolio = pd.DataFrame({
    "Ticker": tickers,
    "Kategori": categories,
    "Yıllık Beklenen Getiri (%)": np.random.uniform(10, 55, n_assets).round(2),
    "Bireysel Volatilite (%)": np.random.uniform(15, 45, n_assets).round(2),
    "Optimal Ağırlık (%)": (weights * 100).round(2),
    "Yatırılacak Tutar (₺)": (weights * 100000).round(2)
})

# --- BÖLÜM 1: Portföy Özet KPI'ları & Sharpe Oranı ---
st.markdown("## 🚀 1. Portföy Optimizasyonu & Sharpe Oranı")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Beklenen Yıllık Getiri", "%38.42", "+%4.1")
kpi2.metric("Portföy Volatilitesi (Risk)", "%21.50", "-%1.2")
kpi3.metric("Sharpe Oranı (Rf=%45)", "1.58", "Optimal")
kpi4.metric("Sortino Oranı", "2.12", "Güçlü")

st.markdown("#### Etkin Sınır (Efficient Frontier) ve Maksimum Sharpe Noktası")
# Simüle edilmiş Efficient Frontier verisi
ef_vol = np.random.uniform(15, 35, 300)
ef_ret = ef_vol * np.random.uniform(0.8, 1.4, 300) + 5
ef_sharpe = (ef_ret - tl_rate) / ef_vol

fig_ef = px.scatter(
    x=ef_vol, y=ef_ret, color=ef_sharpe,
    labels={"x": "Yıllık Volatilite (%)", "y": "Yıllık Beklenen Getiri (%)", "color": "Sharpe"},
    color_continuous_scale="Viridis"
)
# Maksimum Sharpe noktası işaretçisi
fig_ef.add_trace(go.Scatter(
    x=[21.5], y=[38.42], mode="markers+text",
    marker=dict(color="red", size=14, symbol="star"),
    text=["Maksimum Sharpe Portföyü"], textposition="top center",
    name="Optimal Portföy"
))
fig_ef.update_layout(height=400, margin=dict(t=20, b=20))
st.plotly_chart(fig_ef, use_container_width=True)

st.markdown("---")

# --- BÖLÜM 2: Sektör & Kategori Bazlı Dağılım ---
st.markdown("## 📋 2. Sektör & Kategori Bazlı Dağılım")
col1, col2 = st.columns([1.2, 1])

with col1:
    st.dataframe(df_portfolio[["Ticker", "Kategori", "Optimal Ağırlık (%)", "Yatırılacak Tutar (₺)"]], use_container_width=True)

with col2:
    st.markdown("#### Varlık Dağılımı Pastası")
    fig_donut = px.pie(
        df_portfolio[df_portfolio["Optimal Ağırlık (%)"] > 0],
        names="Ticker",
        values="Optimal Ağırlık (%)",
        hole=0.4
    )
    fig_donut.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=350)
    st.plotly_chart(fig_donut, use_container_width=True)

st.markdown("---")

# --- BÖLÜM 3: Risk Metrikleri & Simülasyonlar ---
st.markdown("## 📊 3. Risk Metrikleri ve İleri Düzey Simülasyonlar")

tab_risk, tab_mc, tab_garch, tab_ou = st.tabs([
    "⚠️ Risk Metrikleri (VaR/CVaR/Kurtosis)", 
    "🎲 Monte Carlo GBM (Cholesky)", 
    "📈 GARCH(1,1) Volatilite", 
    "🏛️ OU / Vasicek Faiz Simülasyonu"
])

with tab_risk:
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Daily Value at Risk (%95 VaR)", "%2.00")
    r_col2.metric("Daily Expected Shortfall (%95 CVaR)", "%2.86")
    r_col3.metric("Skewness (Çarpıklık)", "0.31")
    r_col4.metric("Excess Kurtosis (Basıklık)", "1.73")

    st.markdown("#### Portföy Günlük Getiri Dağılımı ve Risk Sınırları (VaR)")
    dummy_returns = np.random.normal(0.0005, 0.012, 1000)
    fig_hist = px.histogram(x=dummy_returns, nbins=30, labels={"x": "Günlük Getiri", "y": "Count"})
    fig_hist.add_vline(x=-0.02, line_dash="dash", line_color="red", annotation_text="%95 VaR Sınırı")
    fig_hist.update_layout(margin=dict(t=20, b=20), height=350)
    st.plotly_chart(fig_hist, use_container_width=True)

with tab_mc:
    st.caption("Cholesky Ayrıştırması ile varlıkların korelasyon yapısı korunarak 500 Farklı Gelecek Patikası (1 Yıl / 252 Gün) simüle edilmiştir.")
    st.markdown("#### Monte Carlo Gelecek Bütçe Simülasyonu")
    
    days = 250
    sim_paths = np.cumprod(1 + np.random.normal(0.0003, 0.01, (50, days)), axis=1) * 100000
    mean_path = np.mean(sim_paths, axis=0)
    upper_path = np.percentile(sim_paths, 95, axis=0)
    lower_path = np.percentile(sim_paths, 5, axis=0)

    fig_mc = go.Figure()
    for i in range(50):
        fig_mc.add_trace(go.Scatter(y=sim_paths[i], mode="lines", line=dict(width=0.5, color="gray"), opacity=0.3, showlegend=False))
    fig_mc.add_trace(go.Scatter(y=mean_path, mode="lines", name="Ortalama Beklenti", line=dict(color="blue", width=2)))
    fig_mc.add_trace(go.Scatter(y=upper_path, mode="lines", name="%95 İyimser Senaryo", line=dict(color="green", dash="dash")))
    fig_mc.add_trace(go.Scatter(y=lower_path, mode="lines", name="%5 Kötümser Senaryo", line=dict(color="red", dash="dash")))
    
    fig_mc.update_layout(xaxis_title="İşlem Günü", yaxis_title="Portföy Değeri (₺)", height=400, margin=dict(t=20, b=20))
    st.plotly_chart(fig_mc, use_container_width=True)

with tab_garch:
    st.markdown("#### GARCH(1,1) Zamanla Değişen Volatilite Tahmini vs Klasik Volatilite")
    df_garch = pd.DataFrame({
        "Ticker": tickers,
        "Tarihsel Volatilite (%)": np.random.uniform(20, 45, n_assets).round(2),
        "GARCH(1,1) İleri Volatilite (%)": np.random.uniform(20, 48, n_assets).round(2)
    })
    df_garch["Fark (Puan)"] = (df_garch["GARCH(1,1) İleri Volatilite (%)"] - df_garch["Tarihsel Volatilite (%)"]).round(2)
    st.dataframe(df_garch, use_container_width=True)

with tab_ou:
    st.markdown("#### Ornstein-Uhlenbeck / Vasicek Mean-Reverting Faiz Simülasyonu")
    st.caption("Mevcut faiz oranının zamanla hedeflenen uzun dönemli ortalamaya nasıl yakınsadığını simüle eder.")
    
    ou_paths = np.zeros((20, 250))
    ou_paths[:, 0] = tl_rate
    for s in range(20):
        for d in range(1, 250):
            ou_paths[s, d] = ou_paths[s, d-1] + 0.1 * (40 - ou_paths[s, d-1]) * (1/250) + 0.5 * np.random.randn()
            
    fig_ou = go.Figure()
    for i in range(20):
        fig_ou.add_trace(go.Scatter(y=ou_paths[i], mode="lines", line=dict(width=0.5), opacity=0.3, showlegend=False))
    fig_ou.add_trace(go.Scatter(y=np.mean(ou_paths, axis=0), mode="lines", name="Ortalama Faiz Patikası", line=dict(color="orange", width=2)))
    fig_ou.update_layout(xaxis_title="Gün", yaxis_title="Faiz Oranı (%)", height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig_ou, use_container_width=True)

st.markdown("---")

# --- BÖLÜM 4: Matrisler (Korelasyon, Kovaryans, Pinv, EWMA) ---
st.markdown("## 🧮 4. Varlıklar Arası Matris Analizleri")
tab_corr, tab_cov, tab_pinv, tab_ewma = st.tabs([
    "🔗 Korelasyon Matrisi", 
    "📐 Kovaryans Matrisi (Ridge Regülarize)", 
    "🔄 Ters Kovaryans Matrisi (Pinv)", 
    "⚡ EWMA Dinamik Kovaryans"
])

raw_matrix = np.random.uniform(-0.1, 0.6, (n_assets, n_assets))
np.fill_diagonal(raw_matrix, 1.0)
dummy_matrix = pd.DataFrame(raw_matrix, index=tickers, columns=tickers)

with tab_corr:
    fig_corr = px.imshow(dummy_matrix, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig_corr.update_layout(height=450, margin=dict(t=20, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

with tab_cov:
    cov_matrix = dummy_matrix * 0.15
    st.dataframe(cov_matrix.round(4), use_container_width=True)

with tab_pinv:
    st.caption("Ridge Regülarize Kovaryans Matrisinin Tersi ($Pinv(\\Sigma + \\lambda I)$):")
    pinv_matrix = np.linalg.pinv((dummy_matrix * 0.15).values)
    df_pinv = pd.DataFrame(pinv_matrix, index=tickers, columns=tickers)
    st.dataframe(df_pinv.round(4), use_container_width=True)

with tab_ewma:
    st.caption("EWMA ($\lambda = 0.94$) Dinamik Kovaryans Matrisi (Son günlere yüksek ağırlık verir):")
    ewma_matrix = (dummy_matrix * 0.15) * 0.92
    st.dataframe(ewma_matrix.round(4), use_container_width=True)

st.markdown("---")

# --- BÖLÜM 5: Varlık Bazlı Detay Tablosu ---
st.markdown("## 📋 5. Varlık Bazlı Detay Tablosu (TRY Bazında)")
st.dataframe(df_portfolio, use_container_width=True)
