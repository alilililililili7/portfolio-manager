import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Advanced Portfolio Allocator",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🏛️ Esnek & Dinamik Portföy Optimizasyon Motoru")

# ==========================================
# SIDEBAR: FORM YAPISI
# ==========================================
with st.sidebar.form(key="portfolio_form"):
    st.header("💰 Bütçe ve Para Birimi")
    currency = st.selectbox(
        "Para Birimi Seçimi", options=["TRY (₺) - TL Bazlı", "USD ($) - Dolar Bazlı"]
    )
    is_tl_mode = "TRY" in currency
    symbol = "₺" if is_tl_mode else "$"

    total_budget = st.number_input(
        f"Toplam Bütçe ({symbol})",
        min_value=100.0,
        value=100000.0 if is_tl_mode else 10000.0,
        step=1000.0 if is_tl_mode else 500.0,
    )

    st.write("---")
    st.subheader("🎯 Varlık Seçimleri")

    # 1. FAİZ SEÇİMİ (Artık Varlık Seçimleri'nin içinde)
    use_rf = st.checkbox("🏦 Risk-Free Faiz Oranını Kullan", value=True)
    default_rf = 45.0 if is_tl_mode else 3.75

    rf_input = st.number_input(
        "Yıllık Faiz / Mevduat Oranı (%)",
        min_value=0.0,
        max_value=100.0,
        value=default_rf,
        step=0.5,
        help="Sharpe Oranı hesabı için kullanılacak risksiz getiri oranı.",
    )

    st.write("---")

    # 2. HİSSE SENETLERİ
    include_stocks = st.checkbox("📈 Hisse Senetleri", value=True)
    stocks_in = st.text_input(
        "Hisseler (Ticker)",
        value="THYAO.IS, EREGL.IS, KCHOL.IS, TUPRS.IS, BIMAS.IS, AKBNK.IS, SISE.IS",
    )

    # 3. ETF'LER
    include_etfs = st.checkbox("🧺 Tematik / Sektörel ETF'ler", value=False)
    etfs_in = st.text_input("ETF'ler (Ticker)", value="SPY, VOO")

    # 4. EMTİA & MADENLER
    include_metals = st.checkbox("🥇 Emtia & Değerli Madenler", value=False)
    metals_in = st.text_input("Maden/Emtia (Ticker)", value="IAU, CPER")

    # 5. EMLAK / GYO
    include_reit = st.checkbox("🏢 Emlak / GYO", value=False)
    reit_in = st.text_input("Emlak/GYO (Ticker)", value="EKGYO.IS")

    # FORM TETİKLEYİCİSİ
    run_opt = st.form_submit_button(
        "🚀 Portföyü Hesapla & Optimize Et", type="primary"
    )

# ==========================================
# MAIN ANALYZER & MATHEMATICAL ENGINE
# ==========================================
selected_assets = {}

if include_stocks and stocks_in:
    for t in stocks_in.split(","):
        if t.strip():
            selected_assets[t.strip().upper()] = "Hisse"

if include_etfs and etfs_in:
    for t in etfs_in.split(","):
        if t.strip():
            selected_assets[t.strip().upper()] = "ETF"

if include_metals and metals_in:
    for t in metals_in.split(","):
        if t.strip():
            selected_assets[t.strip().upper()] = "Emtia/Maden"

if include_reit and reit_in:
    for t in reit_in.split(","):
        if t.strip():
            selected_assets[t.strip().upper()] = "Emlak/GYO"

rf_rate = (rf_input / 100.0) if use_rf else 0.0
FIXED_MIN_WEIGHT = 0.025
tickers = list(selected_assets.keys())

if run_opt:
    if len(tickers) < 2:
        st.error(
            "Matris ve Korelasyon hesaplaması için lütfen en az 2 aktif varlık (Ticker) seçin."
        )
    else:
        try:
            fetch_tickers = tickers.copy()
            has_bist = any(t.endswith(".IS") for t in tickers)
            has_foreign = any(not t.endswith(".IS") for t in tickers)

            need_currency_conversion = (has_bist and has_foreign) or (
                not is_tl_mode and has_bist
            )

            if need_currency_conversion and "USDTRY=X" not in fetch_tickers:
                fetch_tickers.append("USDTRY=X")

            with st.spinner("Piyasa verileri çekiliyor ve analiz ediliyor..."):
                raw_data = yf.download(
                    fetch_tickers, period="3y", progress=False, auto_adjust=True
                )

                if isinstance(raw_data.columns, pd.MultiIndex):
                    df_close = (
                        raw_data["Close"]
                        if "Close" in raw_data.columns.levels[0]
                        else raw_data.xs("Close", axis=1, level=0)
                    )
                else:
                    df_close = (
                        raw_data[["Close"]] if "Close" in raw_data else raw_data
                    )

                if need_currency_conversion and "USDTRY=X" in df_close.columns:
                    usd_try_series = df_close["USDTRY=X"].ffill().bfill()
                    df_close = df_close.drop(columns=["USDTRY=X"])

                    current_usd_try = usd_try_series.iloc[-1]
                    st.sidebar.info(
                        f"💵 Canlı USD/TRY Kuru: **{current_usd_try:.2f} ₺** (Kur Dönüşümü Uygulandı)"
                    )

                    for col in df_close.columns:
                        if col.endswith(".IS"):
                            df_close[col] = df_close[col] / usd_try_series
                else:
                    if "USDTRY=X" in df_close.columns:
                        df_close = df_close.drop(columns=["USDTRY=X"])
                    st.sidebar.success(
                        "📌 Saf TL Modu Aktif (Veriler doğrudan TL fiyatlarıyla hesaplandı)."
                    )

                valid_cols = [t for t in tickers if t in df_close.columns]
                df_close = df_close[valid_cols].dropna(how="any")

                # Log Return & Matrix Math
                log_returns = np.log(df_close / df_close.shift(1)).dropna()
                active_tickers = list(log_returns.columns)
                N = len(active_tickers)

                mean_returns = log_returns.mean() * 252
                cov_matrix = log_returns.cov() * 252
                corr_matrix = log_returns.corr()

                cov_vals = cov_matrix.values + np.eye(N) * 1e-6
                cov_inv = np.linalg.pinv(cov_vals)

                # Sharpe Optimization
                excess_returns = mean_returns.values - rf_rate
                raw_weights = np.dot(cov_inv, excess_returns)

                raw_weights = np.maximum(raw_weights, 0.0)
                if np.sum(raw_weights) > 0:
                    raw_weights = raw_weights / np.sum(raw_weights)
                else:
                    raw_weights = np.ones(N) / N

                weights = np.maximum(raw_weights, FIXED_MIN_WEIGHT)
                excess_w = weights - FIXED_MIN_WEIGHT
                if np.sum(excess_w) > 0:
                    weights = (
                        FIXED_MIN_WEIGHT
                        + (1.0 - N * FIXED_MIN_WEIGHT)
                        * excess_w
                        / np.sum(excess_w)
                    )
                else:
                    weights = np.ones(N) / N

                port_return = np.sum(mean_returns.values * weights)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_vals, weights)))
                sharpe_ratio = (
                    (port_return - rf_rate) / port_vol if port_vol > 0 else 0.0
                )

                allocated_amounts = total_budget * weights

            # ==========================================
            # DISPLAY RESULTS
            # ==========================================
            mode_text = "TL (₺)" if is_tl_mode else "USD ($)"
            st.subheader(f"📌 1. Geniş Portföy Özetleri ({mode_text} Bazlı)")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Toplam Bütçe", f"{symbol}{total_budget:,.2f}")
            m2.metric(
                f"Beklenen Yıllık Getiri ({mode_text})",
                f"%{port_return*100:.2f}",
            )
            m3.metric("Portföy Volatilitesi (Risk)", f"%{port_vol*100:.2f}")
            m4.metric(
                "Sharpe Oranı",
                f"{sharpe_ratio:.2f}",
                help=f"Kıyaslanan Faiz Oranı: %{rf_rate*100:.2f}",
            )

            res_df = pd.DataFrame(
                {
                    "Ticker": active_tickers,
                    "Kategori": [selected_assets[t] for t in active_tickers],
                    f"Yıllık Beklenen Getiri ({mode_text})": [
                        f"%{r*100:.2f}" for r in mean_returns.values
                    ],
                    "Bireysel Volatilite (Risk)": [
                        f"%{v*100:.2f}" for v in np.sqrt(np.diag(cov_vals))
                    ],
                    "Optimal Ağırlık (%)": np.round(weights * 100, 2),
                    f"Yatırılacak Tutar ({symbol})": np.round(
                        allocated_amounts, 2
                    ),
                }
            )

            st.write("---")
            st.subheader("📊 2. Sektör / Kategori Bazlı Dağılım")

            cat_summary = (
                res_df.groupby("Kategori")
                .agg(
                    {
                        "Optimal Ağırlık (%)": "sum",
                        f"Yatırılacak Tutar ({symbol})": "sum",
                    }
                )
                .reset_index()
            )

            c_cat_tbl, c_cat_pie = st.columns([1.8, 1])
            with c_cat_tbl:
                st.dataframe(
                    cat_summary.style.format(
                        {
                            f"Yatırılacak Tutar ({symbol})": f"{symbol}{{:,.2f}}",
                            "Optimal Ağırlık (%)": "%{:.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            with c_cat_pie:
                fig_cat_pie = px.pie(
                    cat_summary,
                    values="Optimal Ağırlık (%)",
                    names="Kategori",
                    hole=0.4,
                )
                st.plotly_chart(fig_cat_pie, use_container_width=True)

            st.write("---")
            st.subheader("📋 3. Varlık Bazlı Detay Tablosu")

            c_tbl, c_pie = st.columns([2.2, 1])
            with c_tbl:
                st.dataframe(
                    res_df.style.format(
                        {
                            f"Yatırılacak Tutar ({symbol})": f"{symbol}{{:,.2f}}",
                            "Optimal Ağırlık (%)": "%{:.2f}",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )
            with c_pie:
                fig_p = px.pie(
                    res_df,
                    values="Optimal Ağırlık (%)",
                    names="Ticker",
                    hole=0.35,
                )
                st.plotly_chart(fig_p, use_container_width=True)

            st.write("---")
            st.subheader("🧮 4. Korelasyon & Kovaryans Matrisleri")

            tab_corr, tab_cov, tab_inv = st.tabs(
                [
                    "🔗 Korelasyon Matrisi",
                    "📐 Kovaryans Matrisi",
                    "🔄 Ters Kovaryans Matrisi",
                ]
            )

            with tab_corr:
                fig_corr = px.imshow(
                    corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

            with tab_cov:
                st.dataframe(cov_matrix.style.format("{:.4f}"))

            with tab_inv:
                df_inv = pd.DataFrame(
                    cov_inv, index=active_tickers, columns=active_tickers
                )
                st.dataframe(df_inv.style.format("{:.4f}"))

        except Exception as e:
            st.error(f"Veri çekme veya matris hesaplama hatası: {str(e)}")
else:
    st.info(
        "👈 Sol menüden seçimlerini tamamlayıp **'🚀 Portföyü Hesapla & Optimize Et'** butonuna basarak analizi başlatabilirsin."
    )
