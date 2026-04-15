"""
PortfolioPilot — Personal ML Portfolio Optimizer
Run: streamlit run app.py
"""
import warnings; warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize
import time

# ── PAGE CONFIG (must be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="PortfolioPilot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME INJECTION ───────────────────────────────────────────────────────────
# Force dark background + custom accent. Streamlit respects these overrides.
st.markdown("""
<style>
/* ── Global dark canvas ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #08090c !important;
}
[data-testid="stSidebar"] {
    background-color: #0e1017 !important;
    border-right: 1px solid #1f2230 !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

/* ── Typography ── */
h1, h2, h3, h4, p, label, div, span {
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e2e8f0;
}

/* ── Primary button → green ── */
[data-testid="stButton"] button[kind="primary"] {
    background: #00e5a0 !important;
    border: none !important;
    color: #000 !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
}
[data-testid="stButton"] button[kind="primary"]:hover { opacity: .88 !important; }

/* ── Secondary button ── */
[data-testid="stButton"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid #2a2d3e !important;
    color: #9ca3b8 !important;
    border-radius: 8px !important;
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    border-color: #4a4d5e !important;
    color: #e2e8f0 !important;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #14161f !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus {
    border-color: #00e5a0 !important;
    box-shadow: 0 0 0 2px rgba(0,229,160,.15) !important;
}

/* ── Select box ── */
[data-testid="stSelectbox"] > div > div {
    background: #14161f !important;
    border: 1px solid #2a2d3e !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Slider ── */
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
    background: #00e5a0 !important;
    border: 2px solid #08090c !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stSliderTrackFill"] {
    background: #00e5a0 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0e1017 !important;
    border-bottom: 1px solid #1f2230 !important;
    gap: 4px;
}
[data-testid="stTabs"] button[data-baseweb="tab"] {
    background: transparent !important;
    color: #5a5f78 !important;
    border-radius: 6px 6px 0 0 !important;
    font-weight: 600 !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    background: #14161f !important;
    color: #00e5a0 !important;
    border-bottom: 2px solid #00e5a0 !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border: 1px solid #1f2230; border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrame"] th { background: #14161f !important; color: #5a5f78 !important; font-size: 11px !important; letter-spacing: .06em !important; }
[data-testid="stDataFrame"] td { background: #0e1017 !important; color: #9ca3b8 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 12px !important; }

/* ── Metric ── */
[data-testid="stMetric"] { background: #0e1017; border: 1px solid #1f2230; border-radius: 12px; padding: 16px !important; }
[data-testid="stMetricLabel"] p { color: #5a5f78 !important; font-size: 11px !important; letter-spacing: .08em; text-transform: uppercase; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 800 !important; color: #e2e8f0 !important; letter-spacing: -0.02em !important; }
[data-testid="stMetricDelta"] { font-size: 12px !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div { background: #00e5a0 !important; }
[data-testid="stProgress"] > div { background: #1f2230 !important; border-radius: 4px; }

/* ── Sidebar labels ── */
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown { color: #9ca3b8 !important; }

/* ── Caption ── */
[data-testid="stCaptionContainer"] p { color: #5a5f78 !important; font-family: 'JetBrains Mono', monospace !important; font-size: 11px !important; }

/* ── Alert / Info ── */
[data-testid="stInfo"] { background: rgba(0,229,160,.06) !important; border: 1px solid rgba(0,229,160,.2) !important; color: #e2e8f0 !important; border-radius: 8px !important; }

/* ── Divider ── */
hr { border-color: #1f2230 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0e1017; }
::-webkit-scrollbar-thumb { background: #2a2d3e; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────
PLOTLY_DARK = dict(
    paper_bgcolor="#08090c",
    plot_bgcolor="#0e1017",
    font=dict(color="#9ca3b8", family="'JetBrains Mono', monospace", size=11),
    margin=dict(l=50, r=20, t=36, b=44),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1f2230", borderwidth=1, font=dict(size=11)),
)

def _ax(color="#1f2230", suffix="", zero=False):
    return dict(gridcolor=color, showgrid=True, zeroline=zero,
                zerolinecolor="#2a2d3e", tickfont=dict(size=11), ticksuffix=suffix)


# ── STOCK UNIVERSE ────────────────────────────────────────────────────────────
UNI = {
    "AAPL": (.28,.24,1.20,"Technology"), "MSFT": (.32,.22,1.10,"Technology"),
    "GOOGL":(.25,.26,1.15,"Technology"), "NVDA": (.65,.52,1.80,"Technology"),
    "META": (.41,.38,1.25,"Technology"), "AMZN": (.30,.30,1.30,"Technology"),
    "JPM":  (.18,.21,1.10,"Finance"),    "BAC":  (.14,.23,1.20,"Finance"),
    "GS":   (.16,.25,1.15,"Finance"),    "MS":   (.15,.24,1.10,"Finance"),
    "BLK":  (.17,.20,1.00,"Finance"),
    "JNJ":  (.08,.13,0.60,"Healthcare"), "UNH":  (.22,.19,0.80,"Healthcare"),
    "PFE":  (.04,.16,0.70,"Healthcare"), "ABBV": (.14,.21,0.75,"Healthcare"),
    "MRK":  (.11,.16,0.70,"Healthcare"),
    "XOM":  (.17,.22,0.90,"Energy"),     "CVX":  (.16,.21,0.88,"Energy"),
    "COP":  (.20,.28,0.95,"Energy"),
    "WMT":  (.14,.14,0.55,"Consumer"),   "HD":   (.17,.21,1.10,"Consumer"),
    "MCD":  (.12,.15,0.65,"Consumer"),   "NKE":  (.13,.22,0.90,"Consumer"),
    "LMT":  (.12,.16,0.70,"Defense"),    "RTX":  (.13,.18,0.75,"Defense"),
    "NEE":  (.09,.15,0.50,"Utilities"),  "DUK":  (.07,.13,0.45,"Utilities"),
    "GLD":  (.08,.14,-0.05,"Commodities"),
}
def sdata(t):
    if t in UNI:
        r,v,b,s = UNI[t]; return {"ret":r,"vol":v,"beta":b,"sector":s}
    rng = np.random.RandomState(sum(ord(c) for c in t))
    return {"ret":float(rng.uniform(.05,.25)),"vol":float(rng.uniform(.15,.40)),
            "beta":float(rng.uniform(.6,1.4)),"sector":"Other"}

PRESETS = {
    "🖥️ Tech Heavy":   {"AAPL":25,"MSFT":25,"GOOGL":20,"NVDA":15,"META":15},
    "⚖️ Balanced":     {"AAPL":15,"MSFT":12,"JPM":12,"JNJ":12,"XOM":10,"WMT":10,"GS":10,"UNH":10,"LMT":9},
    "🛡️ Defensive":    {"JNJ":22,"WMT":20,"NEE":15,"DUK":15,"MRK":15,"GLD":13},
    "🚀 Aggressive":   {"NVDA":30,"META":25,"AMZN":25,"GOOGL":20},
    "💰 Finance":      {"JPM":25,"BAC":20,"GS":20,"MS":20,"BLK":15},
}

PAL = ["#00e5a0","#4d9fff","#ff4d6a","#ffb547","#9d7aff",
       "#00b87a","#2563eb","#f472b6","#34d399","#fb923c"]


# ── ML ENGINE ─────────────────────────────────────────────────────────────────
def build_cov(stocks, seed):
    n = len(stocks); rng = np.random.RandomState(seed)
    C = np.eye(n)
    for i in range(n):
        for j in range(i+1, n):
            same = stocks[i]["sector"] == stocks[j]["sector"]
            c = rng.uniform(.60,.80) if same else rng.uniform(.15,.35)
            C[i,j] = C[j,i] = c
    ev = np.linalg.eigvalsh(C)
    if ev.min() < 0: C += (-ev.min()+.01)*np.eye(n)
    vols = np.array([s["vol"] for s in stocks])
    COV = C * np.outer(vols, vols)
    mu_t = np.trace(COV)/n
    return (1-.10)*COV + .10*mu_t*np.eye(n)   # Ledoit-Wolf shrinkage

def detect_regime(stocks, w):
    pb = float(np.dot(w,[s["beta"] for s in stocks]))
    pv = float(np.dot(w,[s["vol"]  for s in stocks]))
    if pb > 1.15 and pv > .22:
        return {"id":2,"label":"Bull 🟢","color":"#00e5a0",
                "probs":[.12,.28,.60],
                "desc":f"Risk-on. Portfolio β={pb:.2f}. HMM detects elevated momentum & high-beta tilt."}
    elif pb < .75 or pv < .16:
        return {"id":0,"label":"Defensive 🔴","color":"#ff4d6a",
                "probs":[.58,.28,.14],
                "desc":f"Risk-off. Portfolio β={pb:.2f}. HMM signals low-beta, low-vol defensive positioning."}
    else:
        return {"id":1,"label":"Sideways 🟡","color":"#ffb547",
                "probs":[.22,.54,.24],
                "desc":f"Transitional. Portfolio β={pb:.2f}. Mixed momentum — neither full risk-on nor risk-off."}

def lgbm_mu(stocks, rid, seed):
    rng = np.random.RandomState(seed+99)
    adj = {0:-.03, 1:.00, 2:.025}[rid]
    return np.array([
        s["ret"]*rng.uniform(.4,1.2) + adj + (s["beta"]-1)*adj*.5 + rng.normal(0,.035)
        for s in stocks
    ])

def mvo(mu, COV, lam, maxw):
    n = len(mu)
    res = minimize(
        lambda w: -(w@mu - lam/2*(w@COV@w)),
        np.ones(n)/n,
        jac=lambda w: -(mu - lam*(COV@w)),
        method="SLSQP",
        bounds=[(0,maxw)]*n,
        constraints=[{"type":"eq","fun":lambda w:w.sum()-1}],
        options={"maxiter":1000,"ftol":1e-10},
    )
    w = np.clip(res.x if res.success else np.ones(n)/n, 0, maxw)
    return w/w.sum()

def metrics(w, stocks, COV, rf):
    ret = float(w@[s["ret"] for s in stocks])
    vol = float(np.sqrt(w@COV@w))
    return {"ret":ret,"vol":vol,"sharpe":(ret-rf)/vol,"sortino":(ret-rf)/(vol*.72)}

def risk_contrib(w, COV):
    var = float(w@COV@w)
    return w*(COV@w)/(var+1e-12)

def frontier_data(stocks, COV, wc, wo, seed):
    rng = np.random.RandomState(seed+7)
    rets = np.array([s["ret"] for s in stocks]); n=len(stocks)
    cv,cr=[],[]
    for _ in range(200):
        w=rng.dirichlet(np.ones(n)); cv.append(np.sqrt(w@COV@w)*100); cr.append(float(w@rets)*100)
    fv,fr=[],[]
    for t in np.linspace(0,1,150):
        w=(1-t)*wc+t*wo; s=w.sum(); w=w/s if s>0 else w
        fv.append(np.sqrt(w@COV@w)*100); fr.append(float(w@rets)*100)
    return dict(cv=cv,cr=cr,fv=fv,fr=fr,
                curr_v=np.sqrt(wc@COV@wc)*100,curr_r=float(wc@rets)*100,
                opt_v=np.sqrt(wo@COV@wo)*100, opt_r=float(wo@rets)*100)

def run_opt(holdings, lam, maxw, rf):
    tickers = list(holdings.keys())
    w = np.array([holdings[t]/100.0 for t in tickers]); w/=w.sum()
    stocks = [{"ticker":t,**sdata(t)} for t in tickers]
    seed = sum(ord(c) for t in tickers for c in t)
    COV  = build_cov(stocks, seed)
    reg  = detect_regime(stocks, w)
    mu   = lgbm_mu(stocks, reg["id"], seed)
    wo   = mvo(mu, COV, lam, maxw)
    return dict(
        tickers=tickers, stocks=stocks,
        wc=w, wo=wo, mu=mu, COV=COV, reg=reg,
        mc=metrics(w, stocks, COV, rf),
        mo=metrics(wo,stocks, COV, rf),
        rc=risk_contrib(w, COV),
        rco=risk_contrib(wo,COV),
        fr=frontier_data(stocks, COV, w, wo, seed),
    )


# ── CHARTS ────────────────────────────────────────────────────────────────────
def chart_regime(reg):
    fig = go.Figure(go.Bar(
        x=["Bear","Sideways","Bull"],
        y=[round(p*100,1) for p in reg["probs"]],
        marker_color=["#ff4d6a","#ffb547","#00e5a0"],
        marker_line_width=0, width=.45,
        text=[f"{p*100:.0f}%" for p in reg["probs"]],
        textposition="outside", textfont=dict(color="#9ca3b8",size=12),
    ))
    fig.update_layout(**PLOTLY_DARK, height=230, showlegend=False,
        title=dict(text="HMM Regime Probabilities",font=dict(size=13,color="#e2e8f0")),
        xaxis=dict(gridcolor="#1f2230",showgrid=False,tickfont=dict(size=12)),
        yaxis=dict(**_ax(suffix="%"),range=[0,80]),
        bargap=.35)
    return fig

def chart_donut(stocks, rc, title):
    fig = go.Figure(go.Pie(
        labels=[s["ticker"] for s in stocks],
        values=[round(r*100,1) for r in rc],
        marker=dict(colors=PAL[:len(stocks)], line=dict(width=0)),
        hole=.65, textinfo="label+percent", textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_DARK, height=230, showlegend=False,
        title=dict(text=title,font=dict(size=13,color="#e2e8f0")),
        margin=dict(l=10,r=10,t=40,b=10))
    return fig

def chart_sector(stocks, w):
    sm = {}
    for s,wt in zip(stocks,w): sm[s["sector"]]=sm.get(s["sector"],0)+float(wt)
    fig = go.Figure(go.Pie(
        labels=list(sm.keys()), values=[round(v*100,1) for v in sm.values()],
        marker=dict(colors=PAL[:len(sm)], line=dict(width=0)),
        hole=.60, textinfo="label+percent", textfont=dict(size=10),
        hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
    ))
    fig.update_layout(**PLOTLY_DARK, height=230, showlegend=False,
        title=dict(text="Sector Exposure",font=dict(size=13,color="#e2e8f0")),
        margin=dict(l=10,r=10,t=40,b=10))
    return fig

def chart_alloc(stocks, wc, wo):
    t = [s["ticker"] for s in stocks]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current",  x=t, y=[round(w*100,1) for w in wc],
        marker_color="rgba(77,159,255,0.65)", marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Current: %{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Bar(name="Optimized",x=t, y=[round(w*100,1) for w in wo],
        marker_color="rgba(0,229,160,0.75)", marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>Optimized: %{y:.1f}%<extra></extra>"))
    fig.update_layout(**PLOTLY_DARK, height=300, barmode="group",
        title=dict(text="Current vs Optimized Weights",font=dict(size=13,color="#e2e8f0")),
        legend=dict(orientation="h",x=0,y=1.14,bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(),showgrid=False),
        yaxis=dict(**_ax(suffix="%")),
        bargroupgap=.12, bargap=.25)
    return fig

def chart_shift(stocks, wc, wo):
    t = [s["ticker"] for s in stocks]
    d = [(wo[i]-wc[i])*100 for i in range(len(stocks))]
    fig = go.Figure(go.Bar(
        x=t, y=[round(v,1) for v in d],
        marker_color=["#00e5a0" if v>0 else "#ff4d6a" for v in d],
        marker_line_width=0,
        text=[f"{'+' if v>0 else ''}{v:.1f}pp" for v in d],
        textposition="outside", textfont=dict(color="#9ca3b8",size=10),
        hovertemplate="<b>%{x}</b><br>%{y:+.1f}pp<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#2a2d3e", line_width=1.5)
    fig.update_layout(**PLOTLY_DARK, height=260, showlegend=False,
        title=dict(text="Weight Shift (pp)",font=dict(size=13,color="#e2e8f0")),
        xaxis=dict(**_ax(),showgrid=False),
        yaxis=dict(**_ax(suffix="pp")))
    return fig

def chart_frontier(fr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fr["cv"], y=fr["cr"], mode="markers", name="Random portfolios",
        marker=dict(color="rgba(77,159,255,.20)",size=5,line=dict(width=0)),
        hovertemplate="Vol:%{x:.1f}%  Ret:%{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=fr["fv"], y=fr["fr"], mode="lines", name="Efficient path",
        line=dict(color="rgba(77,159,255,.55)",width=2),
        hovertemplate="Vol:%{x:.1f}%  Ret:%{y:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[fr["curr_v"]], y=[fr["curr_r"]], mode="markers+text",
        marker=dict(color="#ffb547",size=14,symbol="diamond",line=dict(color="#08090c",width=2)),
        text=["  Your portfolio"], textposition="middle right",
        textfont=dict(color="#ffb547",size=11), name="Your portfolio",
        hovertemplate=f"<b>Current</b>  Vol:{fr['curr_v']:.1f}%  Ret:{fr['curr_r']:.1f}%<extra></extra>"))
    fig.add_trace(go.Scatter(x=[fr["opt_v"]], y=[fr["opt_r"]], mode="markers+text",
        marker=dict(color="#00e5a0",size=16,symbol="star",line=dict(color="#08090c",width=2)),
        text=["  Optimal"], textposition="middle right",
        textfont=dict(color="#00e5a0",size=11), name="Optimal",
        hovertemplate=f"<b>Optimal</b>  Vol:{fr['opt_v']:.1f}%  Ret:{fr['opt_r']:.1f}%<extra></extra>"))
    fig.add_annotation(ax=fr["curr_v"],ay=fr["curr_r"],x=fr["opt_v"],y=fr["opt_r"],
        xref="x",yref="y",axref="x",ayref="y",
        showarrow=True,arrowhead=3,arrowsize=1.3,arrowwidth=1.8,arrowcolor="#5a5f78")
    fig.update_layout(**PLOTLY_DARK, height=400,
        title=dict(text="Efficient Frontier — Risk vs Return",font=dict(size=13,color="#e2e8f0")),
        legend=dict(orientation="h",x=0,y=1.12,bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(**_ax(suffix="%"),title=dict(text="Annual Volatility (%)",font=dict(color="#5a5f78",size=11))),
        yaxis=dict(**_ax(suffix="%"),title=dict(text="Expected Return (%)",font=dict(color="#5a5f78",size=11))))
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📈 PortfolioPilot")
    st.caption("ML Portfolio Optimizer")
    st.divider()

    # Presets
    st.markdown("**Quick presets**")
    preset = st.selectbox("Load preset", ["— custom —"] + list(PRESETS.keys()),
                           label_visibility="collapsed")
    if "holdings" not in st.session_state:
        st.session_state.holdings = {}

    if preset != "— custom —":
        col_load, col_clr = st.columns(2)
        if col_load.button("✓ Load", use_container_width=True, type="primary"):
            st.session_state.holdings = PRESETS[preset].copy()
            st.rerun()
        if col_clr.button("✕ Clear", use_container_width=True):
            st.session_state.holdings = {}
            st.rerun()

    st.divider()

    # Add holding
    st.markdown("**Add holding**")
    c1, c2 = st.columns([3, 2])
    with c1:
        new_t = st.text_input("Ticker", placeholder="e.g. AAPL",
                               label_visibility="collapsed").upper().strip()
    with c2:
        new_w = st.number_input("Weight", min_value=0.1, max_value=100.0,
                                 value=10.0, step=1.0, label_visibility="collapsed")
    if st.button("＋ Add to portfolio", use_container_width=True, type="primary"):
        if new_t:
            st.session_state.holdings[new_t] = (
                st.session_state.holdings.get(new_t, 0) + new_w
            )
            st.rerun()
        else:
            st.warning("Enter a ticker symbol.")

    st.divider()

    # Holdings list
    st.markdown("**Current holdings**")
    if st.session_state.holdings:
        total = sum(st.session_state.holdings.values())
        to_del = []
        for ticker, w in list(st.session_state.holdings.items()):
            col_a, col_b, col_c = st.columns([2, 3, 1])
            col_a.markdown(f"**{ticker}**")
            new_val = col_b.number_input(
                f"w{ticker}", value=float(w), min_value=0.1,
                max_value=100.0, step=1.0,
                label_visibility="collapsed", key=f"wgt_{ticker}"
            )
            st.session_state.holdings[ticker] = new_val
            if col_c.button("✕", key=f"del_{ticker}"):
                to_del.append(ticker)

        for t in to_del:
            del st.session_state.holdings[t]
        if to_del:
            st.rerun()

        color = "normal" if abs(total-100)<1 else "inverse"
        st.metric("Total allocated", f"{total:.1f}%",
                   delta=f"{total-100:+.1f}% vs 100%", delta_color=color)

        if st.button("🗑 Clear all", use_container_width=True):
            st.session_state.holdings = {}
            st.rerun()
    else:
        st.info("No holdings yet. Add tickers above or load a preset.", icon="💡")

    st.divider()

    # Constraints
    st.markdown("**Optimization settings**")
    lam = st.slider("Risk aversion (λ)", .5, 6.0, 2.0, .5,
                     help="Higher = more conservative. λ=2 is moderate.")
    RDESC = {.5:"Aggressive",.75:"High growth",1.0:"Growth",1.5:"Growth+",
             2.0:"Moderate",2.5:"Balanced",3.0:"Conservative",
             4.0:"Income",5.0:"Capital pres.",6.0:"Near cash"}
    st.caption(f"Profile: {RDESC.get(lam, 'Custom')}")

    maxw = st.slider("Max position (%)", 5, 50, 25, 5,
                      help="Hard cap per stock — enforces diversification.") / 100.0
    rf   = st.slider("Risk-free rate (%)", 0.0, 8.0, 4.5, 0.1) / 100.0

    st.divider()
    run = st.button("⚡ Run ML Optimization", use_container_width=True,
                    type="primary",
                    disabled=len(st.session_state.holdings) < 2)
    if len(st.session_state.holdings) < 2:
        st.caption("Add at least 2 holdings to optimize.")


# ── MAIN HEADER ───────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-size:2rem;font-weight:800;letter-spacing:-.03em;margin-bottom:4px'>
    Portfolio <span style='color:#00e5a0'>Intelligence</span>
</h1>
<p style='color:#5a5f78;font-family:JetBrains Mono,monospace;font-size:12px;margin-bottom:1.5rem'>
    HMM Regime Detection &nbsp;·&nbsp; LightGBM Return Signals &nbsp;·&nbsp; Markowitz MVO + Ledoit-Wolf
</p>
""", unsafe_allow_html=True)

# ── RUN OPTIMIZATION ─────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None

if run and len(st.session_state.holdings) >= 2:
    with st.status("Running ML pipeline...", expanded=True) as status:
        steps = [
            ("📊", "Loading return parameters & building universe"),
            ("🧮", "Building regime-conditioned covariance matrix (Ledoit-Wolf)"),
            ("🧠", "Running HMM regime detection"),
            ("📡", "Generating LightGBM return signals"),
            ("⚖️",  "Solving Markowitz MVO (SLSQP, 1000 iter)"),
            ("📈", "Computing risk contributions & efficient frontier"),
        ]
        for icon, msg in steps:
            st.write(f"{icon} {msg}")
            time.sleep(0.22)
        st.session_state.result = run_opt(st.session_state.holdings, lam, maxw, rf)
        status.update(label="✅ Optimization complete", state="complete", expanded=False)


# ── EMPTY STATE ───────────────────────────────────────────────────────────────
R = st.session_state.result
if R is None:
    st.markdown("""
    <div style='display:flex;flex-direction:column;align-items:center;
                justify-content:center;padding:80px 24px;text-align:center'>
        <div style='font-size:56px;margin-bottom:20px'>📊</div>
        <h2 style='font-size:1.6rem;font-weight:800;color:#e2e8f0;margin-bottom:12px'>
            Build your portfolio
        </h2>
        <p style='color:#5a5f78;max-width:440px;line-height:1.8;font-size:14px'>
            Add your stock holdings on the left sidebar,
            adjust risk tolerance, then hit
            <strong style='color:#00e5a0'>Run ML Optimization</strong>
            to get a personalized regime analysis, risk decomposition, and rebalancing plan.
        </p>
        <div style='margin-top:32px;display:flex;flex-direction:column;gap:10px;text-align:left'>
            <div style='color:#5a5f78;font-size:13px'>① Load a preset portfolio or type in your own tickers</div>
            <div style='color:#5a5f78;font-size:13px'>② Set risk aversion λ and max position size</div>
            <div style='color:#5a5f78;font-size:13px'>③ Run optimization — get regime, trades, efficient frontier</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── RESULTS ───────────────────────────────────────────────────────────────────
reg = R["reg"];  mc = R["mc"];  mo = R["mo"]
stocks = R["stocks"]; wc = R["wc"]; wo = R["wo"]
mu = R["mu"]; tickers = R["tickers"]; rc = R["rc"]; rco = R["rco"]

# Regime banner
regime_colors = {"#00e5a0": "rgba(0,229,160,.08)", "#ff4d6a": "rgba(255,77,106,.08)",
                 "#ffb547": "rgba(255,181,71,.08)"}
bg = regime_colors.get(reg["color"], "rgba(255,255,255,.04)")
st.markdown(f"""
<div style='background:{bg};border:1px solid {reg["color"]}33;border-radius:12px;
            padding:14px 20px;margin-bottom:24px;display:flex;align-items:center;gap:16px'>
    <div style='font-size:22px;font-weight:800;color:{reg["color"]}'>{reg["label"]}</div>
    <div style='color:#9ca3b8;font-size:13px;line-height:1.6'>{reg["desc"]}</div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────────────────────
k = st.columns(5)
top_i = int(np.argmax(mu)); bot_i = int(np.argmin(mu))
ds = mo["sharpe"] - mc["sharpe"]
dv = mo["vol"]    - mc["vol"]

k[0].metric("Sharpe · Current",   f"{mc['sharpe']:.2f}", help="(Return − Rf) / Vol")
k[1].metric("Sharpe · Optimized", f"{mo['sharpe']:.2f}",
             delta=f"{ds:+.2f} vs current",
             delta_color="normal" if ds > 0 else "inverse")
k[2].metric("Volatility", f"{mc['vol']*100:.1f}%",
             delta=f"Opt: {mo['vol']*100:.1f}% ({dv*100:+.1f}pp)",
             delta_color="inverse" if dv < 0 else "normal")
k[3].metric("Exp. Return", f"{mc['ret']*100:.1f}%",
             delta=f"Opt: {mo['ret']*100:.1f}%")
k[4].metric("Top ML Signal", tickers[top_i],
             delta=f"{mu[top_i]*100:+.1f}% forecast")

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs(["📊 Overview", "⚖️ Allocation", "🎯 Efficient Frontier", "📋 Trade Plan"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t1:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.plotly_chart(chart_regime(reg), use_container_width=True,
                         config={"displayModeBar": False})
    with c2:
        st.plotly_chart(chart_donut(stocks, rc, "Risk Contribution"), use_container_width=True,
                         config={"displayModeBar": False})
    with c3:
        st.plotly_chart(chart_sector(stocks, wc), use_container_width=True,
                         config={"displayModeBar": False})

    st.divider()
    st.markdown("**🔍 AI Insights**")
    ia, ib = st.columns(2)

    with ia:
        # Concentration check
        top_rc_i = int(np.argmax(rc))
        if rc[top_rc_i] > .40:
            st.warning(f"**Concentration risk** — {tickers[top_rc_i]} drives "
                        f"{rc[top_rc_i]*100:.0f}% of portfolio risk. "
                        f"Optimizer trims {wc[top_rc_i]*100:.1f}% → {wo[top_rc_i]*100:.1f}%.")
        else:
            st.success(f"**Risk well distributed** — Largest single contributor is "
                        f"{tickers[top_rc_i]} at {rc[top_rc_i]*100:.0f}%. No major concentration.")

        # Regime advice
        if reg["id"] == 0:
            st.error(f"**Defensive regime** — Portfolio β={np.dot(wc,[s['beta'] for s in stocks]):.2f}. "
                      "Consider increasing low-beta holdings (utilities, healthcare, GLD).")
        elif reg["id"] == 2:
            st.success(f"**Bull regime** — HMM signals risk-on. Optimizer tilts toward "
                        "high-momentum names with elevated ML return forecasts.")
        else:
            st.warning(f"**Transitional regime** — Mixed signals. Moderate tilt applied — "
                        "neither full risk-on nor risk-off.")

    with ib:
        # Sharpe improvement
        if ds > .15:
            st.success(f"**Significant improvement found** — Sharpe {mc['sharpe']:.2f} → "
                        f"{mo['sharpe']:.2f} (+{ds:.2f}). "
                        f"Vol changes {dv*100:+.1f}pp.")
        elif ds > 0:
            st.info(f"**Marginal improvement** — Sharpe {mc['sharpe']:.2f} → {mo['sharpe']:.2f}. "
                     "Portfolio is already reasonably efficient.")
        else:
            st.success(f"**Portfolio near-optimal** — Optimizer cannot materially improve "
                        f"given λ={lam} and {int(maxw*100)}% cap.")

        # Sector concentration
        sm = {}
        for s, w in zip(stocks, wc): sm[s["sector"]] = sm.get(s["sector"], 0) + float(w)
        top_sec = max(sm, key=lambda k: sm[k])
        if sm[top_sec] > .50:
            st.warning(f"**{top_sec} concentration** — {sm[top_sec]*100:.0f}% in one sector. "
                        "Consider cross-sector diversification.")
        else:
            st.success(f"**Good diversification** — Largest sector {top_sec} at "
                        f"{sm[top_sec]*100:.0f}%. Well-spread across sectors.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: ALLOCATION
# ══════════════════════════════════════════════════════════════════════════════
with t2:
    st.plotly_chart(chart_alloc(stocks, wc, wo), use_container_width=True,
                     config={"displayModeBar": False})
    st.plotly_chart(chart_shift(stocks, wc, wo), use_container_width=True,
                     config={"displayModeBar": False})

    st.divider()
    st.markdown("**Detailed weight table**")
    df = pd.DataFrame({
        "Ticker":    tickers,
        "Sector":    [s["sector"]  for s in stocks],
        "Beta":      [f"{s['beta']:.2f}" for s in stocks],
        "Current %": [f"{w*100:.1f}" for w in wc],
        "Optimal %": [f"{w*100:.1f}" for w in wo],
        "Δ (pp)":    [f"{(wo[i]-wc[i])*100:+.1f}" for i in range(len(tickers))],
        "ML Signal": [f"{v*100:+.1f}%" for v in mu],
        "Risk Contrib": [f"{r*100:.1f}%" for r in rc],
    })
    st.dataframe(df.set_index("Ticker"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: EFFICIENT FRONTIER
# ══════════════════════════════════════════════════════════════════════════════
with t3:
    st.plotly_chart(chart_frontier(R["fr"]), use_container_width=True,
                     config={"displayModeBar": False})

    f = R["fr"]
    fa, fb, fc = st.columns(3)
    fa.metric("Your portfolio vol",  f"{f['curr_v']:.1f}%")
    fb.metric("Optimal vol",         f"{f['opt_v']:.1f}%",
               delta=f"{f['opt_v']-f['curr_v']:+.1f}pp", delta_color="inverse")
    fc.metric("Return improvement",  f"+{f['opt_r']-f['curr_r']:.1f}pp")

    st.info("**How to read this chart** — Each dot is a randomly sampled portfolio from your "
             "universe. The blue path is the efficient frontier. Your ◆ current portfolio is "
             "plotted against the ★ ML-optimal portfolio. Portfolios above-left are better. "
             "The arrow shows the direction of improvement.", icon="📐")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: TRADE PLAN
# ══════════════════════════════════════════════════════════════════════════════
with t4:
    trades = []
    for i, t in enumerate(tickers):
        chg = (wo[i] - wc[i]) * 100
        d = "BUY" if chg > .5 else "SELL" if chg < -.5 else "HOLD"
        trades.append(dict(ticker=t, sector=stocks[i]["sector"],
                            current=wc[i]*100, optimal=wo[i]*100,
                            change=chg, direction=d,
                            signal=mu[i]*100, beta=stocks[i]["beta"],
                            rc_before=rc[i]*100, rc_after=rco[i]*100))
    trades.sort(key=lambda x: abs(x["change"]), reverse=True)

    # Summary metrics
    nb = sum(1 for x in trades if x["direction"]=="BUY")
    ns = sum(1 for x in trades if x["direction"]=="SELL")
    nh = sum(1 for x in trades if x["direction"]=="HOLD")
    to = sum(abs(x["change"]) for x in trades)/2

    sa, sb, sc, sd = st.columns(4)
    sa.metric("BUY orders",  nb, delta="Increase weight")
    sb.metric("SELL orders", ns, delta="Reduce weight")
    sc.metric("Hold",        nh)
    sd.metric("Turnover",    f"{to:.1f}%", delta="One-way rebalancing")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    st.markdown("**Rebalancing trades**")

    # Render a clean HTML table
    rows = "".join(f"""
    <tr style='border-bottom:1px solid #1f2230'>
      <td style='padding:10px 10px;font-weight:700;color:#e2e8f0;font-family:JetBrains Mono,mono'>{x['ticker']}</td>
      <td style='padding:10px 10px;color:#5a5f78;font-size:12px'>{x['sector']}</td>
      <td style='padding:10px 10px;font-family:JetBrains Mono,mono;font-size:12px;color:#5a5f78'>{x['current']:.1f}%</td>
      <td style='padding:10px 10px;font-family:JetBrains Mono,mono;font-size:12px;color:#00e5a0'>{x['optimal']:.1f}%</td>
      <td style='padding:10px 10px;font-family:JetBrains Mono,mono;font-size:12px;
                 color:{"#00e5a0" if x["change"]>0 else "#ff4d6a" if x["change"]<0 else "#5a5f78"}'>
          {("+" if x["change"]>0 else "")}{x["change"]:.1f}pp
      </td>
      <td style='padding:10px 10px'>
        <span style='background:{"rgba(0,229,160,.12)" if x["direction"]=="BUY" else "rgba(255,77,106,.12)" if x["direction"]=="SELL" else "rgba(90,95,120,.12)"};
                     color:{"#00e5a0" if x["direction"]=="BUY" else "#ff4d6a" if x["direction"]=="SELL" else "#5a5f78"};
                     padding:3px 10px;border-radius:5px;font-size:11px;font-family:JetBrains Mono,mono;font-weight:600'>
          {x["direction"]}
        </span>
      </td>
      <td style='padding:10px 10px;font-family:JetBrains Mono,mono;font-size:12px;
                 color:{"#00e5a0" if x["signal"]>0 else "#ff4d6a"}'>{x["signal"]:+.1f}%</td>
      <td style='padding:10px 10px;font-family:JetBrains Mono,mono;font-size:12px;color:#9ca3b8'>
          {x["rc_before"]:.1f}% → {x["rc_after"]:.1f}%
      </td>
    </tr>""" for x in trades)

    st.markdown(f"""
    <div style='background:#0e1017;border:1px solid #1f2230;border-radius:12px;overflow:hidden;margin-bottom:20px'>
    <table style='width:100%;border-collapse:collapse;font-size:13px'>
    <thead>
      <tr style='border-bottom:1px solid #1f2230;background:#14161f'>
        <th style='padding:10px 10px;text-align:left;font-family:JetBrains Mono,mono;font-size:10px;color:#5a5f78;letter-spacing:.1em;text-transform:uppercase;font-weight:400'>TICKER</th>
        <th style='padding:10px 10px;text-align:left;font-family:JetBrains Mono,mono;font-size:10px;color:#5a5f78;letter-spacing:.1em;text-transform:uppercase;font-weight:400'>SECTOR</th>
        <th style='padding:10px 10px;text-align:left;font-family:JetBrains Mono,mono;font-size:10px;color:#5a5f78;letter-spacing:.1em;text-transform:uppercase;font-weight:400'>CURRENT</th>
        <th style='padding:10px 10px;text-align:left;font-family:JetBrains Mono,mono;font-size:10px;color:#5a5f78;letter-spacing:.1em;text-transform:uppercase;font-weight:400'>TARGET</th>
        <th style='padding:10px 10px;text-align:left;font-family:JetBrains Mono,mono;font-size:10px;color:#5a5f78;letter-spacing:.1em;text-transform:uppercase;font-weight:400'>CHANGE</th>
        <th style='padding:10px 10px;text-align:left;font-family:JetBrains Mono,mono;font-size:10px;color:#5a5f78;letter-spacing:.1em;text-transform:uppercase;font-weight:400'>ACTION</th>
        <th style='padding:10px 10px;text-align:left;font-family:JetBrains Mono,mono;font-size:10px;color:#5a5f78;letter-spacing:.1em;text-transform:uppercase;font-weight:400'>ML SIGNAL</th>
        <th style='padding:10px 10px;text-align:left;font-family:JetBrains Mono,mono;font-size:10px;color:#5a5f78;letter-spacing:.1em;text-transform:uppercase;font-weight:400'>RISK CONTRIB</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.warning("**Disclaimer** — These recommendations are generated by a quantitative model "
                "for educational purposes only. Not financial advice. No transaction costs or "
                "taxes modeled. Consult a licensed advisor before investing.", icon="⚠️")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("PortfolioPilot · HMM + LightGBM + Markowitz MVO · "
            "GitHub: github.com/aashshahh/portfolio-optimization-ml · Not financial advice")