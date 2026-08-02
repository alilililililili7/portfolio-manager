import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Global Portfolio & Asset Manager", page_icon="📈", layout="wide"
)

# Initialize Session State
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "accepted_disclaimer" not in st.session_state:
    st.session_state.accepted_disclaimer = False

# 1. LOGIN SCREEN
if not st.session_state.logged_in:
    st.title("🔐 Login / Register")
    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        if email and password:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Please fill in all fields.")

# 2. LEGAL DISCLAIMER & TERMS OF USE
elif not st.session_state.accepted_disclaimer:
    st.title("⚠️ Legal Disclaimer & Terms of Use")
    st.warning(
        """
    All portfolio optimizations, risk analyses, PE/VC simulations, and mathematical models presented on this platform are strictly for **educational and informational purposes only**.
    
    They do **not** constitute professional **Investment Advice**. The developer and platform cannot be held liable for any financial losses incurred based on these simulations.
    """
    )

    disclaimer_box = st.checkbox(
        "I have read, understood, and accept that this is not investment advice and all responsibility lies with me."
    )

    if st.button("Agree & Enter App"):
        if disclaimer_box:
            st.session_state.accepted_disclaimer = True
            st.rerun()
        else:
            st.error("Please check the box to proceed.")

# 3. CORE APPLICATION (PORTFOLIO ENGINE)
else:
    st.title("🚀 Global Multi-Asset Portfolio Manager")
    st.sidebar.success("Logged in successfully.")

    tab1, tab2, tab3 = st.tabs(
        ["Portfolio Optimization", "Alternative Assets (PE & VC)", "Settings"]
    )

    with tab1:
        st.subheader("Standard Assets (Stocks, ETFs, Bonds, Real Estate)")
        st.info("Matrix calculation modules will be added here.")

    with tab2:
        st.subheader("Venture Capital & Private Equity (Max 10% Rule)")
        st.info(
            "⚠️ High Risk & Illiquid Asset Tracking: PE/VC allocation cannot exceed 10% of total portfolio value."
        )

    with tab3:
        st.subheader("Account & Language Settings")
        st.selectbox("Select Language", ["English", "Türkçe"])
