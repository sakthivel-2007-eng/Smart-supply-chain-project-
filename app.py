import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Smart Supply Chain | Demand Intelligence", page_icon="📦", layout="wide")

st.markdown("""
<style>
    .stApp {background: #f5f7fb;}
    .main-title {font-size: 2.1rem; font-weight: 800; color: #14213d; margin-bottom: 0;}
    .subtitle {color: #64748b; margin-top: .25rem; margin-bottom: 1.2rem;}
    div[data-testid="stMetric"] {background: white; border: 1px solid #e6eaf1; padding: 14px 16px; border-radius: 12px;}
    section[data-testid="stSidebar"] {background: #eef2f8;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📦 Smart Supply Chain — Demand Intelligence</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Big Data Processing and Demand Forecasting for Smart Supply Chains</p>', unsafe_allow_html=True)

def make_sample_data():
    rng = np.random.default_rng(12)
    dates = pd.date_range("2024-01-01", periods=30, freq="MS")
    products = {"Product A": 520, "Product B": 360, "Product C": 245}
    rows = []
    for product, base in products.items():
        for i, date in enumerate(dates):
            seasonal = 1 + 0.13 * np.sin(2 * np.pi * (date.month - 1) / 12)
            trend = 1 + 0.008 * i
            demand = max(40, int(base * seasonal * trend + rng.normal(0, base * 0.055)))
            inventory = max(0, int(demand * rng.uniform(0.65, 1.35)))
            lead = int(rng.integers(3, 13))
            rows.append([date, product, demand, inventory, lead])
    return pd.DataFrame(rows, columns=["Date", "Product", "Demand", "Inventory", "Lead_Time_Days"])

st.sidebar.header("Data & filters")
uploaded = st.sidebar.file_uploader("Upload supply-chain CSV (optional)", type=["csv"])
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        required = {"Date", "Product", "Demand"}
        missing = required - set(df.columns)
        if missing:
            st.error("CSV is missing required columns: " + ", ".join(sorted(missing)))
            st.stop()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Demand"] = pd.to_numeric(df["Demand"], errors="coerce")
        if "Inventory" not in df.columns:
            df["Inventory"] = np.nan
        if "Lead_Time_Days" not in df.columns:
            df["Lead_Time_Days"] = np.nan
        df = df.dropna(subset=["Date", "Product", "Demand"]).copy()
        if df.empty:
            st.error("No valid rows found. Check the Date, Product and Demand columns.")
            st.stop()
    except Exception as e:
        st.error(f"Could not read the CSV: {e}")
        st.stop()
else:
    df = make_sample_data()
    st.sidebar.info("Showing demonstration data. Upload your own CSV to analyze your dataset.")

df = df.sort_values("Date")
products = sorted(df["Product"].astype(str).unique())
selected_product = st.sidebar.selectbox("Select product", products)
periods = st.sidebar.slider("Forecast horizon (months)", 1, 12, 6)

data = df[df["Product"].astype(str) == selected_product].copy().sort_values("Date")
latest_date = data["Date"].max()
total_demand = data["Demand"].sum()
avg_demand = data["Demand"].mean()
latest_demand = data.iloc[-1]["Demand"]
inventory_value = data.iloc[-1]["Inventory"]
lead_value = data.iloc[-1]["Lead_Time_Days"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total recorded demand", f"{total_demand:,.0f}")
c2.metric("Average monthly demand", f"{avg_demand:,.0f}")
c3.metric("Latest month demand", f"{latest_demand:,.0f}")
c4.metric("Latest inventory", "—" if pd.isna(inventory_value) else f"{inventory_value:,.0f}")

st.subheader("Demand trend")
fig = px.line(data, x="Date", y="Demand", markers=True,
              title=f"Historical demand — {selected_product}",
              labels={"Date": "Month", "Demand": "Units"})
fig.update_layout(template="plotly_white", height=350, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fig, use_container_width=True)

# Forecast with a simple trend + annual seasonality regression.
work = data.copy().reset_index(drop=True)
work["t"] = np.arange(len(work))
work["month_sin"] = np.sin(2 * np.pi * work["Date"].dt.month / 12)
work["month_cos"] = np.cos(2 * np.pi * work["Date"].dt.month / 12)
features = work[["t", "month_sin", "month_cos"]]
model = LinearRegression()
model.fit(features, work["Demand"])
future_dates = pd.date_range(latest_date + pd.offsets.MonthBegin(1), periods=periods, freq="MS")
future_t = np.arange(len(work), len(work) + periods)
future_x = pd.DataFrame({
    "t": future_t,
    "month_sin": np.sin(2 * np.pi * future_dates.month / 12),
    "month_cos": np.cos(2 * np.pi * future_dates.month / 12)
})
forecast_values = np.maximum(0, model.predict(future_x))
forecast = pd.DataFrame({"Date": future_dates, "Forecast_Demand": np.round(forecast_values).astype(int)})
st.subheader("Demand forecast")
fc = px.line(forecast, x="Date", y="Forecast_Demand", markers=True,
             title=f"Next {periods} months — model estimate",
             labels={"Date": "Month", "Forecast_Demand": "Forecast units"})
fc.update_layout(template="plotly_white", height=330, margin=dict(l=10, r=10, t=55, b=10))
st.plotly_chart(fc, use_container_width=True)

left, right = st.columns([1.1, 1])
with left:
    st.subheader("Forecast values")
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.download_button("Download forecast CSV", forecast.to_csv(index=False).encode("utf-8"),
                       file_name=f"{selected_product.lower().replace(' ', '_')}_forecast.csv",
                       mime="text/csv")
with right:
    st.subheader("Inventory & replenishment view")
    if data["Inventory"].notna().any():
        last_inventory = data.dropna(subset=["Inventory"]).iloc[-1]["Inventory"]
        next_demand = forecast.iloc[0]["Forecast_Demand"]
        gap = next_demand - last_inventory
        st.metric("Estimated next-month stock gap", f"{gap:,.0f} units",
                  delta="Potential shortage" if gap > 0 else "Stock may be sufficient",
                  delta_color="inverse" if gap > 0 else "normal")
        st.caption("A positive gap means forecast demand is higher than the latest recorded inventory. This is a planning indicator, not an automatic purchase order.")
    else:
        st.info("Add an Inventory column to your CSV to view a stock-gap indicator.")
    if data["Lead_Time_Days"].notna().any():
        st.metric("Latest recorded supplier lead time",
                  f"{data['Lead_Time_Days'].dropna().iloc[-1]:.0f} days")
    else:
        st.info("Add a Lead_Time_Days column to include supplier lead-time analysis.")

with st.expander("CSV format and project methodology"):
    st.markdown("""
    **Required columns:** `Date`, `Product`, `Demand`

    **Optional columns:** `Inventory`, `Lead_Time_Days`

    **Example:**
    | Date | Product | Demand | Inventory | Lead_Time_Days |
    |---|---|---:|---:|---:|
    | 2025-01-01 | Product A | 520 | 600 | 7 |
    | 2025-02-01 | Product A | 545 | 580 | 8 |

    **Processing workflow:** data upload → validation and cleaning → product-wise filtering
    → historical trend analysis → regression-based demand forecast → inventory planning indicator.

    The included dataset is synthetic demonstration data. Forecasts are educational estimates;
    validate the model with real historical data before making business decisions.
    """)

st.caption("Capstone prototype • Built with Python, Streamlit, Pandas, Scikit-learn and Plotly")
