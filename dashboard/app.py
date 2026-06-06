"""
KYC Risk Scoring & AML Transaction Monitoring Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import sys
import os

# --- Path setup ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

DB_PATH     = os.path.join(ROOT, "database", "kyc_aml.db")
MODEL_DIR   = os.path.join(ROOT, "models")
DATA_DIR    = os.path.join(ROOT, "data")

# --- Color Palette ---
COLORS = {
    "HIGH":   "#EF4444",
    "MEDIUM": "#F59E0B",
    "LOW":    "#10B981",
    "bg":     "#0F172A",
    "card":   "#1E293B",
    "accent": "#3B82F6",
    "text":   "#F1F5F9",
}

st.set_page_config(
    page_title="KYC/AML Monitoring | JPMorgan",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0F172A;
    color: #F1F5F9;
}
.main { background-color: #0F172A; }

.metric-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 6px 0;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
    line-height: 1;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #94A3B8;
    margin-top: 4px;
}
.risk-badge-HIGH   { background:#EF4444; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.risk-badge-MEDIUM { background:#F59E0B; color:#000; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }
.risk-badge-LOW    { background:#10B981; color:#fff; padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.2em;
    color: #3B82F6;
    border-bottom: 1px solid #1E3A5F;
    padding-bottom: 6px;
    margin-bottom: 16px;
}
.alert-row {
    background: #1E293B;
    border-left: 3px solid #EF4444;
    padding: 10px 16px;
    margin: 6px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
}
stTabs [data-baseweb="tab"] { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)


# --- DB helper ---
@st.cache_data(ttl=60)
def query(sql, params=()):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    return df


# ═══════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 KYC/AML System")
    st.markdown("*JPMorgan Chase · Analyst Dashboard*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Executive Overview", "👤 KYC Customer Risk", "🚨 AML Alert Center", "🔍 Transaction Deep-Dive", "➕ New Customer Onboarding"],
        label_visibility="collapsed"
    )
    st.divider()
    st.markdown('<p style="font-size:0.7rem;color:#475569;">Model: GBM + Isolation Forest<br>KYC Model: Random Forest<br>DB: SQLite · Records: 5,000</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ═══════════════════════════════════════════════════════
if page == "📊 Executive Overview":
    st.markdown("# Executive Overview")
    st.markdown('<p class="section-header">Real-Time KYC/AML Monitoring · Q4 2024</p>', unsafe_allow_html=True)

    # KPIs
    total_custs  = query("SELECT COUNT(*) as n FROM customers").iloc[0]["n"]
    high_risk_c  = query("SELECT COUNT(*) as n FROM customers WHERE risk_tier='HIGH'").iloc[0]["n"]
    medium_risk_c= query("SELECT COUNT(*) as n FROM customers WHERE risk_tier='MEDIUM'").iloc[0]["n"]
    total_txns   = query("SELECT COUNT(*) as n FROM transactions").iloc[0]["n"]
    flagged_txns = query("SELECT COUNT(*) as n FROM transactions WHERE aml_flagged=1").iloc[0]["n"]
    open_alerts  = query("SELECT COUNT(*) as n FROM aml_alerts WHERE resolution='OPEN'").iloc[0]["n"]
    pep_count    = query("SELECT COUNT(*) as n FROM customers WHERE is_pep=1").iloc[0]["n"]
    sanctions    = query("SELECT COUNT(*) as n FROM customers WHERE sanctions_hit=1").iloc[0]["n"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#3B82F6">{total_custs:,}</div>
            <div class="metric-label">Total Customers</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#EF4444">{high_risk_c}</div>
            <div class="metric-label">High Risk Customers</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#F59E0B">{flagged_txns:,}</div>
            <div class="metric-label">AML Flagged Transactions</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#EF4444">{open_alerts:,}</div>
            <div class="metric-label">Open AML Alerts</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#A78BFA">{pep_count}</div>
            <div class="metric-label">PEP Customers</div></div>""", unsafe_allow_html=True)
    with c6:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#EF4444">{sanctions}</div>
            <div class="metric-label">Sanctions Hits</div></div>""", unsafe_allow_html=True)
    with c7:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#10B981">{total_txns:,}</div>
            <div class="metric-label">Total Transactions</div></div>""", unsafe_allow_html=True)
    with c8:
        flag_rate = flagged_txns / total_txns * 100
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#F59E0B">{flag_rate:.1f}%</div>
            <div class="metric-label">AML Flag Rate</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Customer Risk Distribution")
        risk_dist = query("SELECT risk_tier, COUNT(*) as count FROM customers GROUP BY risk_tier")
        fig = px.pie(risk_dist, values="count", names="risk_tier",
                     color="risk_tier",
                     color_discrete_map={"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"},
                     hole=0.55)
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#F1F5F9", legend=dict(font=dict(color="#F1F5F9")),
                          height=320, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### AML Alert Types")
        alert_types = query("""
            SELECT alert_type, COUNT(*) as count
            FROM aml_alerts WHERE alert_type != 'NONE'
            GROUP BY alert_type ORDER BY count DESC
        """)
        fig2 = px.bar(alert_types, x="count", y="alert_type", orientation="h",
                      color="count", color_continuous_scale=["#1E3A5F", "#EF4444"])
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#F1F5F9", height=320,
                           margin=dict(t=10,b=10), yaxis_title="", xaxis_title="Alert Count",
                           coloraxis_showscale=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Transaction volume over time
    st.markdown("#### Transaction Volume & AML Flags — Last 12 Months")
    txn_time = query("""
        SELECT substr(txn_date,1,7) as month,
               COUNT(*) as total,
               SUM(aml_flagged) as flagged
        FROM transactions
        GROUP BY month ORDER BY month
    """)
    fig3 = go.Figure()
    fig3.add_trace(go.Bar(x=txn_time["month"], y=txn_time["total"],
                          name="Total", marker_color="#1E3A5F"))
    fig3.add_trace(go.Scatter(x=txn_time["month"], y=txn_time["flagged"],
                              name="AML Flagged", line=dict(color="#EF4444", width=2.5),
                              mode="lines+markers"))
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font_color="#F1F5F9", height=300, margin=dict(t=10,b=10),
                       legend=dict(font=dict(color="#F1F5F9")))
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 2 — KYC CUSTOMER RISK
# ═══════════════════════════════════════════════════════
elif page == "👤 KYC Customer Risk":
    st.markdown("# KYC Customer Risk")
    st.markdown('<p class="section-header">Know Your Customer · Risk Scoring Engine</p>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
    with col_f1:
        search = st.text_input("🔍 Search customer name or ID", placeholder="e.g. CUST00042 or John")
    with col_f2:
        risk_filter = st.selectbox("Risk Tier", ["ALL", "HIGH", "MEDIUM", "LOW"])
    with col_f3:
        pep_filter = st.checkbox("PEP Only", value=False)

    sql = "SELECT * FROM customers WHERE 1=1"
    params = []
    if search:
        sql += " AND (customer_id LIKE ? OR name LIKE ?)"
        params += [f"%{search}%", f"%{search}%"]
    if risk_filter != "ALL":
        sql += " AND risk_tier = ?"
        params.append(risk_filter)
    if pep_filter:
        sql += " AND is_pep = 1"
    sql += " ORDER BY kyc_score DESC LIMIT 200"

    custs = query(sql, params)

    # KYC Score histogram
    col_h1, col_h2 = st.columns([2, 1])
    with col_h1:
        fig_hist = px.histogram(custs, x="kyc_score", nbins=30, color="risk_tier",
                                color_discrete_map={"HIGH":"#EF4444","MEDIUM":"#F59E0B","LOW":"#10B981"},
                                title="KYC Score Distribution")
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#F1F5F9", height=250, margin=dict(t=40,b=10))
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_h2:
        st.markdown("#### Risk Breakdown")
        for tier in ["HIGH", "MEDIUM", "LOW"]:
            n = len(custs[custs["risk_tier"] == tier])
            color = COLORS[tier]
            st.markdown(f'<span class="risk-badge-{tier}">{tier}</span> &nbsp; **{n}** customers', unsafe_allow_html=True)
            st.markdown("")

    # Customer table
    st.markdown("#### Customer Risk Register")
    display = custs[["customer_id","name","country","occupation","kyc_score","risk_tier","is_pep","sanctions_hit","account_balance"]].copy()
    display.columns = ["ID","Name","Country","Occupation","KYC Score","Risk","PEP","Sanctions","Balance"]

    def color_risk(val):
        colors = {"HIGH":"background-color:#7F1D1D;color:#FCA5A5",
                  "MEDIUM":"background-color:#78350F;color:#FDE68A",
                  "LOW":"background-color:#064E3B;color:#6EE7B7"}
        return colors.get(val, "")

    styled = display.style.map(color_risk, subset=["Risk"]) \
                          .format({"KYC Score": "{:.1f}", "Balance": "${:,.0f}"})
    st.dataframe(styled, use_container_width=True, height=380)

    # Customer detail
    st.markdown("---")
    st.markdown("#### Customer Deep-Dive")
    cust_id = st.selectbox("Select Customer ID", custs["customer_id"].tolist())
    if cust_id:
        cust = custs[custs["customer_id"] == cust_id].iloc[0]
        cust_txns = query("SELECT * FROM transactions WHERE customer_id=? ORDER BY txn_date DESC LIMIT 20",
                          (cust_id,))

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            **Name:** {cust['name']}  
            **Country:** {cust['country']} `{cust['country_risk']}`  
            **Occupation:** {cust['occupation']}  
            **Member Since:** {cust['onboarding_date']}
            """)
        with c2:
            st.markdown(f"""
            **KYC Score:** `{cust['kyc_score']}`  
            **Risk Tier:** <span class="risk-badge-{cust['risk_tier']}">{cust['risk_tier']}</span>  
            **PEP Status:** {'⚠️ Yes' if cust['is_pep'] else '✅ No'}  
            **Sanctions Hit:** {'🚨 Yes' if cust['sanctions_hit'] else '✅ No'}
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            **Account Balance:** ${cust['account_balance']:,.2f}  
            **Adverse Media:** {'⚠️ Yes' if cust['adverse_media'] else '✅ No'}  
            **Total Transactions:** {len(cust_txns)}  
            **Flagged Transactions:** {cust_txns['aml_flagged'].sum()}
            """)

        if not cust_txns.empty:
            fig_txn = px.scatter(cust_txns, x="txn_date", y="amount", color="aml_flagged",
                                 color_discrete_map={0:"#10B981", 1:"#EF4444"},
                                 size="amount", hover_data=["txn_type","alert_type"],
                                 title=f"Transaction History — {cust_id}")
            fig_txn.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#F1F5F9", height=280, margin=dict(t=40,b=10))
            st.plotly_chart(fig_txn, use_container_width=True)


# ═══════════════════════════════════════════════════════
# PAGE 3 — AML ALERT CENTER
# ═══════════════════════════════════════════════════════
elif page == "🚨 AML Alert Center":
    st.markdown("# AML Alert Center")
    st.markdown('<p class="section-header">Anti-Money Laundering · Real-Time Alert Management</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        alert_type_filter = st.selectbox("Alert Type", ["ALL","STRUCTURING","HIGH_RISK_JURISDICTION",
                                                          "SANCTIONS_HIT","LARGE_TRANSACTION","UNUSUAL_ACTIVITY"])
    with c2:
        resolution_filter = st.selectbox("Status", ["ALL", "OPEN", "CLEARED", "FLAGGED"])

    sql = """
        SELECT a.alert_id, a.txn_id, a.customer_id, c.name,
               a.alert_type, a.alert_date, a.amount, a.risk_tier, a.resolution
        FROM aml_alerts a JOIN customers c ON a.customer_id = c.customer_id
        WHERE 1=1
    """
    params = []
    if alert_type_filter != "ALL":
        sql += " AND a.alert_type = ?"; params.append(alert_type_filter)
    if resolution_filter != "ALL":
        sql += " AND a.resolution = ?"; params.append(resolution_filter)
    sql += " ORDER BY a.amount DESC LIMIT 300"

    alerts = query(sql, params)

    # Summary row
    s1, s2, s3, s4, s5 = st.columns(5)
    for col, atype, color in zip(
        [s1,s2,s3,s4,s5],
        ["STRUCTURING","HIGH_RISK_JURISDICTION","SANCTIONS_HIT","LARGE_TRANSACTION","UNUSUAL_ACTIVITY"],
        ["#A78BFA","#EF4444","#F59E0B","#3B82F6","#EC4899"]
    ):
        n = len(alerts[alerts["alert_type"] == atype])
        col.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:{color};font-size:1.4rem">{n}</div>
            <div class="metric-label">{atype.replace('_',' ')}</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Alert heatmap by type x risk tier
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.markdown("#### Alerts by Type & Risk Tier")
        heat_data = query("""
            SELECT alert_type, risk_tier, COUNT(*) as count
            FROM aml_alerts WHERE alert_type!='NONE'
            GROUP BY alert_type, risk_tier
        """)
        if not heat_data.empty:
            pivot = heat_data.pivot(index="alert_type", columns="risk_tier", values="count").fillna(0)
            fig_heat = px.imshow(pivot, color_continuous_scale=["#0F172A","#1E3A5F","#EF4444"],
                                 aspect="auto")
            fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#F1F5F9", height=280, margin=dict(t=20,b=10))
            st.plotly_chart(fig_heat, use_container_width=True)

    with col_v2:
        st.markdown("#### Alert Amount Distribution")
        if not alerts.empty:
            fig_box = px.box(alerts, x="alert_type", y="amount", color="alert_type",
                             color_discrete_sequence=["#EF4444","#F59E0B","#A78BFA","#3B82F6","#10B981"])
            fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#F1F5F9", height=280, margin=dict(t=20,b=10),
                                  showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(fig_box, use_container_width=True)

    # Alert table
    st.markdown("#### Active Alerts")
    if not alerts.empty:
        display = alerts.copy()
        display["amount"] = display["amount"].map("${:,.2f}".format)
        st.dataframe(display, use_container_width=True, height=400)
    else:
        st.info("No alerts match the selected filters.")


# ═══════════════════════════════════════════════════════
# PAGE 4 — TRANSACTION DEEP-DIVE
# ═══════════════════════════════════════════════════════
elif page == "🔍 Transaction Deep-Dive":
    st.markdown("# Transaction Deep-Dive")
    st.markdown('<p class="section-header">Granular Transaction Analysis · Pattern Detection</p>', unsafe_allow_html=True)

    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        txn_type_filter = st.selectbox("Transaction Type",
            ["ALL","WIRE_TRANSFER","CASH_DEPOSIT","CASH_WITHDRAWAL","ONLINE_TRANSFER","CHECK","ACH"])
    with col_f2:
        amount_min = st.number_input("Min Amount ($)", value=0, step=1000)
    with col_f3:
        show_flagged_only = st.checkbox("AML Flagged Only", value=False)

    sql = "SELECT t.*, c.name, c.risk_tier FROM transactions t JOIN customers c ON t.customer_id=c.customer_id WHERE 1=1"
    params = []
    if txn_type_filter != "ALL":
        sql += " AND t.txn_type=?"; params.append(txn_type_filter)
    if amount_min > 0:
        sql += " AND t.amount>=?"; params.append(amount_min)
    if show_flagged_only:
        sql += " AND t.aml_flagged=1"
    sql += " ORDER BY t.amount DESC LIMIT 500"

    txns = query(sql, params)

    # Stats row
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Transactions", f"{len(txns):,}")
    s2.metric("Total Volume", f"${txns['amount'].sum():,.0f}")
    s3.metric("Avg Amount", f"${txns['amount'].mean():,.0f}")
    s4.metric("AML Flagged", f"{txns['aml_flagged'].sum()} ({txns['aml_flagged'].mean()*100:.1f}%)")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### Amount by Transaction Type")
        fig_type = px.box(txns, x="txn_type", y="amount", color="aml_flagged",
                          color_discrete_map={0:"#10B981", 1:"#EF4444"})
        fig_type.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#F1F5F9", height=300, margin=dict(t=20,b=10),
                               xaxis_tickangle=-20)
        st.plotly_chart(fig_type, use_container_width=True)

    with col_c2:
        st.markdown("#### Transaction Hour Heatmap")
        txns["hour"] = pd.to_datetime(txns["txn_date"]).dt.hour
        txns["dow"]  = pd.to_datetime(txns["txn_date"]).dt.day_name()
        hour_heat = txns.groupby(["dow","hour"]).size().reset_index(name="count")
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        hour_heat["dow"] = pd.Categorical(hour_heat["dow"], categories=day_order, ordered=True)
        hour_heat = hour_heat.sort_values("dow")
        pivot_h = hour_heat.pivot(index="dow", columns="hour", values="count").fillna(0)
        fig_hh = px.imshow(pivot_h, color_continuous_scale=["#0F172A","#1E3A5F","#3B82F6"],
                           aspect="auto", labels=dict(x="Hour of Day"))
        fig_hh.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="#F1F5F9", height=300, margin=dict(t=20,b=10))
        st.plotly_chart(fig_hh, use_container_width=True)

    st.markdown("#### Cross-Border Transaction Risk Matrix")
    cb = txns[txns["is_cross_border"]==1].groupby(["dest_risk","aml_flagged"]).size().reset_index(name="count")
    if not cb.empty:
        fig_cb = px.bar(cb, x="dest_risk", y="count", color="aml_flagged",
                        color_discrete_map={0:"#10B981",1:"#EF4444"},
                        barmode="group",
                        labels={"dest_risk":"Destination Risk","count":"Transactions"})
        fig_cb.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font_color="#F1F5F9", height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig_cb, use_container_width=True)

    st.markdown("#### Transaction Records")
    show_cols = ["txn_id","customer_id","name","txn_date","amount","txn_type",
                 "dest_country","is_cross_border","aml_flagged","alert_type","risk_tier"]
    st.dataframe(txns[[c for c in show_cols if c in txns.columns]], use_container_width=True, height=350)


# ═══════════════════════════════════════════════════════
# PAGE 5 — NEW CUSTOMER ONBOARDING
# ═══════════════════════════════════════════════════════
elif page == "➕ New Customer Onboarding":
    st.markdown("# New Customer Onboarding")
    st.markdown('<p class="section-header">KYC Screening · Instant Risk Assessment</p>', unsafe_allow_html=True)

    HIGH_RISK   = ["Iran","North Korea","Myanmar","Syria","Sudan","Cuba","Venezuela"]
    MEDIUM_RISK = ["Pakistan","Nigeria","Kenya","Bangladesh","Egypt","Ukraine","Russia"]
    LOW_RISK    = ["United States","United Kingdom","Germany","France","Japan","Canada","Australia","Singapore","India","Netherlands"]
    ALL_COUNTRIES = sorted(HIGH_RISK + MEDIUM_RISK + LOW_RISK)

    OCCUPATION_RISK_MAP = {
        "Politician": 0.9, "Government Official": 0.85, "Arms Dealer": 0.95,
        "Cryptocurrency Trader": 0.7, "Lawyer": 0.5, "Doctor": 0.2,
        "Engineer": 0.15, "Teacher": 0.1, "Retail Worker": 0.1,
        "Banker": 0.3, "Business Owner": 0.45, "Journalist": 0.25,
    }

    with st.form("onboarding_form"):
        st.markdown("#### Customer Profile")
        c1, c2 = st.columns(2)
        with c1:
            name       = st.text_input("Full Name", placeholder="Jane Smith")
            email      = st.text_input("Email Address")
            country    = st.selectbox("Country of Residence", ALL_COUNTRIES)
            occupation = st.selectbox("Occupation", list(OCCUPATION_RISK_MAP.keys()))
        with c2:
            phone      = st.text_input("Phone Number")
            balance    = st.number_input("Initial Deposit ($)", min_value=100.0, value=5000.0, step=100.0)
            is_pep     = st.checkbox("Politically Exposed Person (PEP)?")
            sanctions  = st.checkbox("Sanctions List Match?")
            adverse    = st.checkbox("Adverse Media Detected?")

        submitted = st.form_submit_button("🔍 Run KYC Screening", use_container_width=True)

    if submitted and name:
        # Country risk
        if country in HIGH_RISK:   c_risk, c_score = "HIGH", 0.9
        elif country in MEDIUM_RISK: c_risk, c_score = "MEDIUM", 0.5
        else:                      c_risk, c_score = "LOW", 0.1

        occ_risk = OCCUPATION_RISK_MAP.get(occupation, 0.3)

        kyc_score = (
            c_score * 30 +
            occ_risk * 25 +
            (25 if is_pep else 0) +
            (15 if sanctions else 0) +
            (10 if adverse else 0)
        )
        kyc_score = round(min(max(kyc_score, 0), 100), 2)

        if kyc_score >= 65:   risk_tier = "HIGH"
        elif kyc_score >= 35: risk_tier = "MEDIUM"
        else:                 risk_tier = "LOW"

        st.markdown("---")
        st.markdown("## 📋 KYC Screening Result")

        res_color = COLORS[risk_tier]
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1E293B,#0F172A);border:2px solid {res_color};
                    border-radius:16px;padding:28px 32px;margin:12px 0;">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <div style="font-size:1.5rem;font-weight:700">{name}</div>
                    <div style="color:#94A3B8;font-size:0.85rem">{country} · {occupation}</div>
                </div>
                <div style="text-align:right">
                    <div style="font-size:3rem;font-weight:700;font-family:'IBM Plex Mono',monospace;color:{res_color}">{kyc_score}</div>
                    <div style="color:{res_color};font-weight:600;font-size:0.9rem">{risk_tier} RISK</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.markdown("**Risk Factors**")
            st.markdown(f"🌍 Country Risk: `{c_risk}` (+{c_score*30:.0f} pts)")
            st.markdown(f"💼 Occupation Risk: (+{occ_risk*25:.0f} pts)")
            if is_pep:     st.markdown("⚠️ PEP Status: +25 pts")
            if sanctions:  st.markdown("🚨 Sanctions Hit: +15 pts")
            if adverse:    st.markdown("📰 Adverse Media: +10 pts")
        with col_r2:
            st.markdown("**Onboarding Decision**")
            if risk_tier == "LOW":
                st.success("✅ APPROVED — Standard onboarding")
            elif risk_tier == "MEDIUM":
                st.warning("⚠️ ENHANCED DUE DILIGENCE required")
            else:
                st.error("🚫 ESCALATE to Compliance Officer")
        with col_r3:
            st.markdown("**Required Actions**")
            if risk_tier == "LOW":
                st.markdown("- Standard ID verification\n- Basic source of funds\n- Annual review")
            elif risk_tier == "MEDIUM":
                st.markdown("- Enhanced ID verification\n- Source of wealth docs\n- 6-month review\n- Transaction monitoring")
            else:
                st.markdown("- Senior compliance approval\n- Full EDD report\n- Source of funds proof\n- Ongoing monitoring\n- Monthly review")

        # Score gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=kyc_score,
            domain={"x":[0,1],"y":[0,1]},
            gauge={
                "axis": {"range":[0,100], "tickcolor":"#94A3B8"},
                "bar":  {"color": res_color, "thickness": 0.25},
                "steps":[
                    {"range":[0,35],  "color":"#064E3B"},
                    {"range":[35,65], "color":"#78350F"},
                    {"range":[65,100],"color":"#7F1D1D"},
                ],
                "threshold":{"line":{"color":res_color,"width":4},"value":kyc_score},
            },
            number={"font":{"color":"#F1F5F9","family":"IBM Plex Mono"}},
        ))
        fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#F1F5F9",
                                height=280, margin=dict(t=20,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)
