import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from datetime import date

# --------------------------- APP CONFIG ---------------------------
st.set_page_config(
    page_title="NEXUS | Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------- DESIGN SYSTEM ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
:root { --ink:#17243b; --muted:#718096; --line:#e7edf5; --bg:#f5f7fb; --blue:#315efb; --teal:#11a99a; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:var(--bg); color:var(--ink); }
.block-container { padding-top:1.5rem; padding-bottom:2.5rem; max-width:1500px; }
section[data-testid="stSidebar"] { background:#101c32; border-right:1px solid #24334c; }
section[data-testid="stSidebar"] * { color:#e8eef8 !important; }
section[data-testid="stSidebar"] .stRadio label { padding:8px 10px; border-radius:9px; }
section[data-testid="stSidebar"] .stRadio label:hover { background:#20314d; }
.brand {font-family:'Manrope',sans-serif;font-size:1.2rem;font-weight:800;letter-spacing:.02em;color:#fff;padding:8px 0 2px;}
.brand-sub {font-size:.72rem;color:#9fb0c8;margin-bottom:1.5rem;}
.eyebrow {font-size:.72rem;font-weight:800;letter-spacing:.13em;text-transform:uppercase;color:#64748b;margin-bottom:.35rem;}
.page-title {font-family:'Manrope',sans-serif;font-size:2rem;font-weight:800;letter-spacing:-.04em;color:#14213a;margin:0;}
.page-sub {color:#718096;margin:.25rem 0 1.25rem;font-size:.94rem;}
.metric-card {background:#fff;border:1px solid var(--line);border-radius:15px;padding:17px 18px;min-height:112px;box-shadow:0 3px 12px rgba(25,45,80,.035);}
.metric-label {font-size:.78rem;color:#718096;font-weight:600;}
.metric-value {font-family:'Manrope',sans-serif;font-size:1.75rem;font-weight:800;color:#17243b;margin-top:5px;}
.metric-foot {font-size:.72rem;color:#8b98aa;margin-top:3px;}
.panel {background:#fff;border:1px solid var(--line);border-radius:15px;padding:18px 20px;box-shadow:0 3px 12px rgba(25,45,80,.025);}
.panel-title {font-family:'Manrope',sans-serif;font-weight:800;font-size:1rem;color:#17243b;margin-bottom:2px;}
.panel-sub {font-size:.78rem;color:#8491a3;margin-bottom:12px;}
.pill {display:inline-block;background:#e9f7f4;color:#078678;padding:4px 9px;border-radius:99px;font-size:.7rem;font-weight:700;}
div[data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:12px;overflow:hidden;}
.stButton>button {border-radius:10px;font-weight:700;min-height:2.6rem;}
.stDownloadButton>button {border-radius:10px;font-weight:700;}
div[data-testid="stMetric"] {background:white;border:1px solid var(--line);padding:14px 16px;border-radius:13px;}
hr {border-color:var(--line);}
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# --------------------------- DATA ---------------------------
@st.cache_data
def sample_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=42, freq="MS")
    products = [
        ("P-1001", "Smart Sensor", "Electronics", 480, 260, 8, "Northstar Components"),
        ("P-1002", "Control Unit", "Electronics", 310, 190, 12, "Vertex Electronics"),
        ("P-1003", "Packaging Kit", "Packaging", 720, 410, 5, "PackRight Industries"),
        ("P-1004", "Power Module", "Electronics", 255, 170, 14, "Vertex Electronics"),
        ("P-1005", "Safety Assembly", "Industrial", 390, 220, 9, "Apex Industrial")
    ]
    rows=[]
    for pid, name, cat, base, stock, lead, supplier in products:
        for i, dt in enumerate(dates):
            seasonal = 1 + .16*np.sin(2*np.pi*(dt.month-1)/12)
            trend = 1 + .004*i
            demand = max(20, int(base*seasonal*trend + rng.normal(0, base*.07)))
            inv = max(0, int(stock + rng.normal(0, 35) - i*1.2))
            rows.append([dt,pid,name,cat,demand,inv,lead,supplier,round(float(rng.uniform(18,145)),2)])
    return pd.DataFrame(rows, columns=["Date","Product_ID","Product","Category","Demand","Inventory","Lead_Time_Days","Supplier","Unit_Cost"])

def normalize_data(raw):
    df=raw.copy()
    # Accept a few common column spellings.
    aliases={"date":"Date","product":"Product","product_name":"Product","demand":"Demand",
             "inventory":"Inventory","stock":"Inventory","category":"Category",
             "product_id":"Product_ID","lead_time_days":"Lead_Time_Days",
             "lead_time":"Lead_Time_Days","supplier":"Supplier","unit_cost":"Unit_Cost"}
    rename={}
    for c in df.columns:
        key=str(c).strip().lower().replace(" ","_")
        if key in aliases: rename[c]=aliases[key]
    df=df.rename(columns=rename)
    required={"Date","Product","Demand"}
    missing=required-set(df.columns)
    if missing: raise ValueError("Required columns missing: "+", ".join(sorted(missing)))
    df["Date"]=pd.to_datetime(df["Date"],errors="coerce")
    df["Demand"]=pd.to_numeric(df["Demand"],errors="coerce")
    for col, default in [("Inventory",np.nan),("Lead_Time_Days",np.nan),("Unit_Cost",np.nan)]:
        if col not in df: df[col]=default
        df[col]=pd.to_numeric(df[col],errors="coerce")
    for col, default in [("Category","Uncategorized"),("Supplier","Unassigned"),("Product_ID","")]:
        if col not in df: df[col]=default
        df[col]=df[col].fillna(default).astype(str)
    df["Product"]=df["Product"].astype(str)
    df=df.dropna(subset=["Date","Demand","Product"])
    df=df[df["Demand"]>=0]
    if df.empty: raise ValueError("No valid data rows found.")
    return df.sort_values("Date")

def forecast_product(data, horizon, method):
    d=data.groupby(pd.Grouper(key="Date",freq="MS"))["Demand"].sum().reset_index().dropna()
    d=d.sort_values("Date").reset_index(drop=True)
    if len(d)<3:
        raise ValueError("At least three monthly observations are needed to generate a forecast.")
    d["t"]=np.arange(len(d))
    d["sin"]=np.sin(2*np.pi*d["Date"].dt.month/12)
    d["cos"]=np.cos(2*np.pi*d["Date"].dt.month/12)
    X=d[["t","sin","cos"]]
    if method=="Random Forest":
        model=RandomForestRegressor(n_estimators=180,max_depth=5,random_state=42)
    else:
        model=LinearRegression()
    model.fit(X,d["Demand"])
    future_dates=pd.date_range(d["Date"].max()+pd.offsets.MonthBegin(1),periods=horizon,freq="MS")
    ft=np.arange(len(d),len(d)+horizon)
    FX=pd.DataFrame({"t":ft,"sin":np.sin(2*np.pi*future_dates.month/12),
                     "cos":np.cos(2*np.pi*future_dates.month/12)})
    pred=np.maximum(0,model.predict(FX))
    result=pd.DataFrame({"Date":future_dates,"Forecast_Demand":np.round(pred).astype(int)})
    # Holdout error indicator: train on all-but-last few points, if enough data.
    test_mae=None
    if len(d)>=8:
        split=max(5,len(d)-4)
        eval_model=RandomForestRegressor(n_estimators=180,max_depth=5,random_state=42) if method=="Random Forest" else LinearRegression()
        eval_model.fit(X.iloc[:split],d["Demand"].iloc[:split])
        test_mae=mean_absolute_error(d["Demand"].iloc[split:],eval_model.predict(X.iloc[split:]))
    return result,d,test_mae

def metric(label,value,foot=""):
    st.markdown(f"""<div class="metric-card"><div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>""",unsafe_allow_html=True)

# --------------------------- SIDEBAR ---------------------------
with st.sidebar:
    st.markdown('<div class="brand">◈ NEXUS</div><div class="brand-sub">SUPPLY CHAIN INTELLIGENCE</div>',unsafe_allow_html=True)
    st.markdown(" ")
    page=st.radio("WORKSPACE",["Executive Overview","Demand Forecasting","Inventory Planning","Supplier Insights","Data Workspace"],label_visibility="visible")
    st.markdown("---")
    st.markdown("**DATA SOURCE**")
    uploaded=st.file_uploader("Upload CSV dataset",type=["csv"],help="Required: Date, Product, Demand. Other columns are optional.")
    st.caption("No file uploaded? The app uses a built-in sample dataset.")
    st.markdown("---")
    st.caption("CAPSTONE PROTOTYPE · v1.0")

if uploaded:
    try:
        raw=pd.read_csv(uploaded)
        df=normalize_data(raw)
        data_source="Uploaded dataset"
    except Exception as e:
        st.error(f"Could not process uploaded file: {e}")
        st.stop()
else:
    df=sample_data()
    data_source="Sample demonstration dataset"

# Filters
all_products=sorted(df["Product"].unique())
all_categories=sorted(df["Category"].unique())
with st.sidebar:
    selected_categories=st.multiselect("Category filter",all_categories,default=all_categories)
    selected_products=st.multiselect("Product filter",all_products,default=all_products)
filtered=df[df["Category"].isin(selected_categories)&df["Product"].isin(selected_products)].copy()
if filtered.empty:
    st.warning("No records match the selected filters. Adjust the sidebar filters.")
    st.stop()

# Header
st.markdown('<div class="eyebrow">OPERATIONS CONTROL CENTER</div>',unsafe_allow_html=True)
st.markdown(f'<h1 class="page-title">{page}</h1>',unsafe_allow_html=True)
st.markdown(f'<p class="page-sub">A unified view of demand, inventory and supplier performance · {data_source}</p>',unsafe_allow_html=True)

# --------------------------- EXECUTIVE OVERVIEW ---------------------------
if page=="Executive Overview":
    total=int(filtered["Demand"].sum())
    avg=float(filtered.groupby(pd.Grouper(key="Date",freq="MS"))["Demand"].sum().mean())
    latest=filtered[filtered["Date"]==filtered["Date"].max()]["Demand"].sum()
    products_n=filtered["Product"].nunique()
    a,b,c,d=st.columns(4)
    with a: metric("TOTAL DEMAND",f"{total:,.0f}","Units across selected records")
    with b: metric("AVG. MONTHLY DEMAND",f"{avg:,.0f}","Units per month")
    with c: metric("LATEST PERIOD",f"{latest:,.0f}","Units in most recent month")
    with d: metric("ACTIVE PRODUCTS",f"{products_n:,}","Products in current view")
    st.write("")
    l,r=st.columns([1.55,1])
    with l:
        st.markdown('<div class="panel"><div class="panel-title">Demand performance</div><div class="panel-sub">Monthly demand across selected products</div>',unsafe_allow_html=True)
        trend=filtered.groupby(pd.Grouper(key="Date",freq="MS"))["Demand"].sum().reset_index()
        fig=px.area(trend,x="Date",y="Demand",template="plotly_white")
        fig.update_traces(line_color="#315efb",fillcolor="rgba(49,94,251,.12)")
        fig.update_layout(height=330,margin=dict(l=5,r=5,t=10,b=5),xaxis_title="",yaxis_title="Units",hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    with r:
        st.markdown('<div class="panel"><div class="panel-title">Demand by category</div><div class="panel-sub">Share of total recorded demand</div>',unsafe_allow_html=True)
        cat=filtered.groupby("Category",as_index=False)["Demand"].sum()
        fig=px.pie(cat,names="Category",values="Demand",hole=.62,template="plotly_white")
        fig.update_layout(height=330,margin=dict(l=0,r=0,t=5,b=5),legend_title_text="")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('</div>',unsafe_allow_html=True)
    st.write("")
    st.markdown("#### Product demand ranking")
    rank=filtered.groupby(["Product","Category"],as_index=False)["Demand"].sum().sort_values("Demand",ascending=False)
    rank["Demand share (%)"]=(rank["Demand"]/rank["Demand"].sum()*100).round(1)
    st.dataframe(rank,use_container_width=True,hide_index=True)
    st.download_button("Download overview data",filtered.to_csv(index=False).encode(), "supply_chain_overview.csv","text/csv")

# --------------------------- DEMAND FORECASTING ---------------------------
elif page=="Demand Forecasting":
    col1,col2,col3=st.columns([1,1,1])
    with col1: horizon=st.selectbox("Forecast horizon", [3,6,9,12],index=1,format_func=lambda x:f"{x} months")
    with col2: method=st.selectbox("Forecasting model",["Linear Regression","Random Forest"])
    with col3: product=st.selectbox("Forecast product",sorted(filtered["Product"].unique()))
    product_data=filtered[filtered["Product"]==product]
    try:
        forecast,hist,mae=forecast_product(product_data,horizon,method)
        st.write("")
        total_forecast=int(forecast["Forecast_Demand"].sum())
        avg_forecast=float(forecast["Forecast_Demand"].mean())
        k1,k2,k3=st.columns(3)
        with k1: metric("FORECAST TOTAL",f"{total_forecast:,.0f}","Units over selected horizon")
        with k2: metric("MONTHLY FORECAST",f"{avg_forecast:,.0f}","Average predicted units")
        with k3: metric("HOLDOUT MAE","—" if mae is None else f"{mae:,.1f}","Historical validation error · units")
        st.write("")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=hist["Date"],y=hist["Demand"],mode="lines+markers",name="Historical demand",line=dict(color="#315efb",width=3)))
        fig.add_trace(go.Scatter(x=forecast["Date"],y=forecast["Forecast_Demand"],mode="lines+markers",name="Forecast",line=dict(color="#11a99a",width=3,dash="dash")))
        fig.update_layout(template="plotly_white",height=390,margin=dict(l=10,r=10,t=20,b=10),xaxis_title="",yaxis_title="Units",legend=dict(orientation="h",y=1.12))
        st.plotly_chart(fig,use_container_width=True)
        l,r=st.columns([1,1])
        with l:
            st.markdown("#### Forecast schedule")
            st.dataframe(forecast,use_container_width=True,hide_index=True)
        with r:
            st.markdown("#### Planning interpretation")
            recent=float(hist.tail(3)["Demand"].mean())
            diff=(avg_forecast-recent)/recent*100 if recent else 0
            if diff>5: st.success(f"Forecast demand is approximately {diff:.1f}% above the recent 3-month average. Review replenishment capacity.")
            elif diff< -5: st.info(f"Forecast demand is approximately {abs(diff):.1f}% below the recent 3-month average. Review excess-stock exposure.")
            else: st.info("Forecast demand is broadly in line with the recent 3-month average.")
            st.caption("Model output is an estimate. Compare it with operational knowledge before acting.")
        st.download_button("Download forecast report",forecast.to_csv(index=False).encode(),"demand_forecast.csv","text/csv")
    except Exception as e:
        st.error(str(e))

# --------------------------- INVENTORY PLANNING ---------------------------
elif page=="Inventory Planning":
    latest_date=filtered["Date"].max()
    latest=filtered[filtered["Date"]==latest_date].groupby(["Product","Category","Supplier"],as_index=False).agg(
        Inventory=("Inventory","mean"),Demand=("Demand","mean"),Lead_Time_Days=("Lead_Time_Days","mean"))
    latest["Daily demand estimate"]=latest["Demand"]/30
    latest["Reorder point"]=latest["Daily demand estimate"]*latest["Lead_Time_Days"]
    latest["Stock status"]=np.where(latest["Inventory"].isna(),"No inventory data",
        np.where(latest["Inventory"]<latest["Reorder point"],"Reorder","Healthy"))
    latest["Suggested replenishment"]=np.where(latest["Inventory"].notna(),
        np.maximum(0,np.ceil(latest["Reorder point"]*1.5-latest["Inventory"])),np.nan)
    a,b,c=st.columns(3)
    with a: metric("PRODUCTS TO REVIEW",str((latest["Stock status"]=="Reorder").sum()),"Below estimated reorder point")
    with b: metric("PRODUCTS MONITORED",str(len(latest)),"Latest period product records")
    with c: metric("LATEST DATA DATE",latest_date.strftime("%b %Y"),"Most recent month in dataset")
    st.write("")
    st.markdown("#### Replenishment recommendations")
    st.caption("Reorder point = estimated daily demand × supplier lead time. Suggested replenishment targets 1.5× reorder point.")
    show=latest[["Product","Category","Supplier","Inventory","Reorder point","Suggested replenishment","Stock status"]].copy()
    for col in ["Inventory","Reorder point","Suggested replenishment"]: show[col]=show[col].round(0)
    st.dataframe(show,use_container_width=True,hide_index=True)
    st.download_button("Download replenishment plan",show.to_csv(index=False).encode(),"replenishment_plan.csv","text/csv")
    st.caption("This is a planning aid based on simplified assumptions. Confirm safety stock, MOQ and purchase constraints before ordering.")

# --------------------------- SUPPLIER INSIGHTS ---------------------------
elif page=="Supplier Insights":
    sup=filtered.groupby("Supplier",as_index=False).agg(
        Total_Demand=("Demand","sum"),Avg_Lead_Time=("Lead_Time_Days","mean"),
        Product_Count=("Product","nunique"),Avg_Unit_Cost=("Unit_Cost","mean"))
    sup["Avg_Lead_Time"]=sup["Avg_Lead_Time"].round(1)
    sup["Avg_Unit_Cost"]=sup["Avg_Unit_Cost"].round(2)
    a,b,c=st.columns(3)
    with a: metric("SUPPLIERS",str(len(sup)),"In selected data")
    with b: metric("AVG. LEAD TIME",f"{filtered['Lead_Time_Days'].mean():.1f} days" if filtered["Lead_Time_Days"].notna().any() else "—","Across available records")
    with c: metric("SUPPLIER DEMAND",f"{sup['Total_Demand'].sum():,.0f}","Units associated with suppliers")
    st.write("")
    l,r=st.columns([1.3,1])
    with l:
        st.markdown("#### Demand by supplier")
        fig=px.bar(sup.sort_values("Total_Demand",ascending=True),x="Total_Demand",y="Supplier",orientation="h",template="plotly_white",color="Total_Demand",color_continuous_scale=["#c9d7ff","#315efb"])
        fig.update_layout(height=350,margin=dict(l=5,r=5,t=10,b=5),coloraxis_showscale=False,xaxis_title="Demand units",yaxis_title="")
        st.plotly_chart(fig,use_container_width=True)
    with r:
        st.markdown("#### Supplier lead time")
        lead=sup.dropna(subset=["Avg_Lead_Time"])
        fig=px.bar(lead,x="Supplier",y="Avg_Lead_Time",template="plotly_white",color_discrete_sequence=["#11a99a"])
        fig.update_layout(height=350,margin=dict(l=5,r=5,t=10,b=5),xaxis_title="",yaxis_title="Days")
        st.plotly_chart(fig,use_container_width=True)
    st.markdown("#### Supplier summary")
    st.dataframe(sup,use_container_width=True,hide_index=True)
    st.download_button("Download supplier summary",sup.to_csv(index=False).encode(),"supplier_summary.csv","text/csv")

# --------------------------- DATA WORKSPACE ---------------------------
elif page=="Data Workspace":
    st.markdown("#### Dataset health")
    total_rows=len(df)
    missing=int(df[["Date","Product","Demand","Inventory","Lead_Time_Days"]].isna().sum().sum())
    x1,x2,x3=st.columns(3)
    with x1: metric("DATA ROWS",f"{total_rows:,}","Records loaded")
    with x2: metric("PRODUCTS",str(df["Product"].nunique()),"Unique product names")
    with x3: metric("MISSING KEY VALUES",f"{missing:,}","Across key analysis columns")
    st.write("")
    st.markdown("#### Data preview")
    st.dataframe(filtered.head(500),use_container_width=True,hide_index=True)
    st.download_button("Download filtered dataset",filtered.to_csv(index=False).encode(),"filtered_supply_chain_data.csv","text/csv")
    st.markdown("#### Upload template")
    template=pd.DataFrame([
        ["2025-01-01","P-1001","Smart Sensor","Electronics",520,600,7,"Northstar Components",45.50],
        ["2025-02-01","P-1001","Smart Sensor","Electronics",545,580,7,"Northstar Components",45.50]
    ],columns=["Date","Product_ID","Product","Category","Demand","Inventory","Lead_Time_Days","Supplier","Unit_Cost"])
    st.download_button("Download CSV template",template.to_csv(index=False).encode(),"supply_chain_template.csv","text/csv")
    st.info("For reliable monthly forecasting, provide at least 3 months of history per product. More history generally improves model evaluation.")

st.markdown("---")
st.markdown("<div style='display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;color:#8290a4;font-size:.75rem'><span>© 2026 NEXUS · Smart Supply Chain Intelligence</span><span>Prototype for academic demonstration · Verify insights before operational use</span></div>",unsafe_allow_html=True)
