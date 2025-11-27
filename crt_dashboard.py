import os, time, math, random
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.express as px

# =========================
# Config
# =========================
REFRESH_SECONDS = 60
ROLLING_WINDOW = 20
MC_PATHS = 2000
MC_DAYS = 30
LAM = 0.25  # Volatility dampening

# =========================
# Opportunities
# =========================
DEFAULT_OPPS = [
    {"name":"Apple (AAPL)","type":"Stock","A":8,"I":9,"S":7,"ticker":"AAPL"},
    {"name":"Tesla (TSLA)","type":"Stock","A":7,"I":8,"S":6,"ticker":"TSLA"},
    {"name":"Ethereum (ETH)","type":"Crypto","A":6,"I":7,"S":5,"ticker":"ETH-USD"},
    {"name":"SPY ETF","type":"ETF","A":7,"I":7,"S":6,"ticker":"SPY"},
    {"name":"Gold (GC=F)","type":"Commodity","A":6,"I":6,"S":6,"ticker":"GC=F"},
    {"name":"Crude Oil (CL=F)","type":"Commodity","A":6,"I":6,"S":6,"ticker":"CL=F"},
    {"name":"EURUSD","type":"Forex","A":5,"I":6,"S":5,"ticker":"EURUSD=X"},
    {"name":"Freelance Writing","type":"Side Hustle","A":9,"I":8,"S":9,"expected_return":20},
    {"name":"E-commerce Store","type":"Business","A":7,"I":8,"S":7,"expected_return":18},
    {"name":"NFT Drop","type":"Speculative","A":5,"I":6,"S":5,"expected_return":30},
    {"name":"YouTube/TikTok","type":"Content","A":8,"I":7,"S":8,"expected_return":25},
    {"name":"AI Service","type":"Business","A":7,"I":8,"S":7,"expected_return":22},
    {"name":"Domain Flipping","type":"Digital Asset","A":6,"I":7,"S":6,"expected_return":12},
    {"name":"Affiliate Marketing","type":"Business","A":8,"I":8,"S":7,"expected_return":19}
]

# =========================
# Helpers
# =========================
@st.cache_data(ttl=REFRESH_SECONDS)
def fetch_history(ticker):
    try:
        df = yf.Ticker(ticker).history(period="90d", interval="1d")
        return df if not df.empty else pd.DataFrame()
    except:
        return pd.DataFrame()

def compute_vol(df):
    if df.empty or "Close" not in df: return 5.0
    returns = df["Close"].pct_change().dropna()
    return max(returns.std()*100, 1.0)

def compute_expected(df):
    if df.empty or "Close" not in df: return 0.0
    close = df["Close"]
    if len(close) <= ROLLING_WINDOW: return 0.0
    return (close.iloc[-1]/close.iloc[-ROLLING_WINDOW]-1)*100.0

def crt_score(A,I,S,E,vol_norm):
    base = (A*I*S)/max(E,1e-6)
    return base*(1 - LAM*vol_norm)

def normalize(v, lo=1, hi=10):
    v = max(lo,min(hi,v))
    return (v-lo)/(hi-lo)

def monte_carlo(S0, mu_pct, sigma_pct):
    if S0 <=0 or sigma_pct<=0: return {"best":0,"worst":0,"median":0}
    mu = mu_pct/100/MC_DAYS
    sigma = sigma_pct/100/math.sqrt(MC_DAYS)
    sims = []
    for _ in range(MC_PATHS):
        price=S0
        for _d in range(MC_DAYS):
            z=np.random.normal()
            price*=(1+mu+sigma*z)
        sims.append((price-S0)/S0*100)
    sims=np.array(sims)
    return {"best":float(np.percentile(sims,95)),
            "worst":float(np.percentile(sims,5)),
            "median":float(np.percentile(sims,50))}

def log_top5(df):
    ts=datetime.utcnow().isoformat()
    lines=[f"[{ts}] === Top 5 CRT Snapshot ==="]
    for _,r in df.iterrows():
        lines.append(f"{r['name']} | CRT: {r['CRT_adj']:.2f} | Ret: {r['expected_return_pct']:.2f}% | Risk: {r['risk_pct']:.2f}% | CRT/Risk: {r['CRT_per_risk']:.2f}")
    with open("CRT_Ranking.txt","a",encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CRT Money Dashboard", layout="wide")
st.title("CRT Money Dashboard — Multi-Market, Overnight Ready")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    classes = st.multiselect("Asset Classes", ["Stock","ETF","Crypto","Commodity","Forex","Side Hustle","Business","Speculative","Content","Digital Asset"],
                             default=["Stock","ETF","Crypto","Commodity","Forex","Side Hustle","Business"])
    lam = st.slider("Volatility Dampening λ",0.0,1.0,LAM,0.05)
    interval = st.number_input("Refresh (sec)",10,300,REFRESH_SECONDS)

# Filter opportunities
opps=[o.copy() for o in DEFAULT_OPPS if o["type"] in classes]
rows=[]
for o in opps:
    ticker=o.get("ticker")
    df=pd.DataFrame()
    E=3.0; risk=10.0; exp_ret=o.get("expected_return",10); S0=None; vol_norm=0.2
    if ticker:
        df=fetch_history(ticker)
        E=compute_vol(df)
        risk=compute_vol(df)
        exp_ret=compute_expected(df)
        try: S0=float(df["Close"].iloc[-1])
        except: S0=None
        vol_norm=normalize(risk)
    crt_adj=crt_score(o["A"],o["I"],o["S"],E,vol_norm)
    mc={"best":None,"median":None,"worst":None}
    if S0: mc=monte_carlo(S0,mu_pct=exp_ret,sigma_pct=risk)
    rows.append({"name":o["name"],"type":o["type"],"ticker":ticker or "",
                 "A":o["A"],"I":o["I"],"S":o["S"],"E":E,
                 "risk_pct":risk,"expected_return_pct":exp_ret,
                 "CRT_adj":crt_adj,"CRT_per_risk":crt_adj/max(risk,1e-6),
                 "MC_best_%":mc["best"],"MC_median_%":mc["median"],"MC_worst_%":mc["worst"]})
df_all=pd.DataFrame(rows)
df_ranked=df_all.sort_values(["CRT_adj","expected_return_pct"],ascending=[False,False]).reset_index(drop=True)
top5=df_ranked.head(5)

# Log top 5
log_top5(top5)

# Alerts
ALERT_CRT_RISK=1.2; ALERT_RET=0.5
alerts=df_ranked[(df_ranked["CRT_per_risk"]>=ALERT_CRT_RISK)&(df_ranked["expected_return_pct"]>=ALERT_RET)]

# Layout
col1,col2=st.columns([1.2,1])
with col1:
    st.subheader("Risk vs Expected Return")
    fig=px.scatter(df_ranked,x="risk_pct",y="expected_return_pct",color="type",size="CRT_adj",
                   hover_data=["name","ticker","CRT_adj","CRT_per_risk"],labels={"risk_pct":"Risk %","expected_return_pct":"Return %","CRT_adj":"CRT"},
                   title="Risk vs Expected Return (size = CRT)")
    st.plotly_chart(fig,use_container_width=True)
with col2:
    st.subheader("Top 5 Opportunities")
    st.dataframe(top5[["name","type","ticker","CRT_adj","CRT_per_risk","expected_return_pct","risk_pct","MC_best_%","MC_median_%","MC_worst_%"]],use_container_width=True)
st.subheader("All Opportunities")
st.dataframe(df_ranked,use_container_width=True)
with st.expander("Alerts"):
    if alerts.empty: st.write("No alerts triggered.")
    else:
        for _,r in alerts.iterrows():
            st.write(f"ALERT: {r['name']} — CRT/Risk {r['CRT_per_risk']:.2f}, Expected {r['expected_return_pct']:.2f}%")
st.caption(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Log: CRT_Ranking.txt")
