import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Service Desk Analytics", layout="wide")

# --- DATA CONNECTION ---
# We use a function to cache the connection so it doesn't reopen on every reload
@st.cache_resource
def get_connection():
    # Connect to the local file we uploaded to GitHub
    conn = duckdb.connect('service_desk.duckdb', read_only=True)
    return conn

con = get_connection()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Executive Pulse (VP View)", "Technician Bottlenecks (Ops View)"])
st.sidebar.markdown("---")
st.sidebar.caption("Data Source: DuckDB Local Warehouse")

# =========================================================
# PAGE 1: EXECUTIVE PULSE
# =========================================================
if page == "Executive Pulse (VP View)":
    st.title("📊 Executive Service Desk Pulse")
    st.markdown("High-level health check for the VP of Services.")

    # 1. Fetch KPI Data
    df_pulse = con.sql("SELECT * FROM vw_kpi_executive_pulse ORDER BY year DESC, month_number DESC").df()
    current = df_pulse.iloc[0]
    previous = df_pulse.iloc[1] if len(df_pulse) > 1 else current

    # 2. KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Volume", f"{current['total_ticket_volume']}", 
                  delta=f"{current['total_ticket_volume'] - previous['total_ticket_volume']}")
    with col2:
        st.metric("MTTR (Hours)", f"{current['mttr_hours']}h", 
                  delta=f"{round(current['mttr_hours'] - previous['mttr_hours'], 1)}h", delta_color="inverse")
    with col3:
        st.metric("SLA Breach Rate", f"{current['sla_breach_rate']}%", 
                  delta=f"{round(current['sla_breach_rate'] - previous['sla_breach_rate'], 1)}%", delta_color="inverse")
    with col4:
        st.metric("CSAT Score", f"{current['avg_csat']}/5.0", 
                  delta=f"{round(current['avg_csat'] - previous['avg_csat'], 2)}")

    st.divider()

    # 3. Charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📉 Backlog Growth Trend")
        df_backlog = con.sql("SELECT * FROM vw_kpi_backlog_history WHERE full_date >= CURRENT_DATE - INTERVAL 90 DAY").df()
        fig_backlog = px.line(df_backlog, x='full_date', y='total_backlog', title='Active Backlog (Last 90 Days)')
        fig_backlog.update_traces(line_color='#FF4B4B')
        st.plotly_chart(fig_backlog, use_container_width=True)

    with c2:
        st.subheader("📈 Resolution Efficiency")
        df_trend = df_pulse.sort_values(by=['year', 'month_number'])
        fig_eff = px.bar(df_trend, x='month_name', y='fcr_rate', title='First Contact Resolution % (Monthly)', text='fcr_rate')
        fig_eff.update_traces(marker_color='#1F77B4')
        st.plotly_chart(fig_eff, use_container_width=True)

# =========================================================
# PAGE 2: TECHNICIAN BOTTLENECKS
# =========================================================
elif page == "Technician Bottlenecks (Ops View)":
    st.title("🕵️ Technician Performance Matrix")
    st.markdown("Identify outliers, training opportunities, and bottlenecks.")

    # 1. Fetch Data
    df_tech = con.sql("SELECT * FROM vw_kpi_tech_performance").df()

    # 2. Scatter Plot
    st.subheader("The Efficiency Matrix")
    st.caption("X-Axis: Volume | Y-Axis: CSAT | Size: Re-open Rate")
    
    fig_scatter = px.scatter(
        df_tech, 
        x="tickets_resolved", 
        y="avg_csat", 
        size="reopen_rate", 
        color="role_level",
        hover_name="full_name",
        hover_data=["avg_handle_time_mins", "reopen_rate"],
        text="full_name",
        title="High Volume vs. High Quality",
        height=500
    )
    fig_scatter.update_traces(textposition='top center')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 3. Drill Down Grid
    st.divider()
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("🚨 The 'Bottleneck' List")
        st.caption("Technicians with High AHT or High Re-open Rates.")
        df_display = df_tech[['full_name', 'role_level', 'avg_handle_time_mins', 'reopen_rate', 'tickets_resolved']]
        st.dataframe(
            df_display.style.background_gradient(subset=['avg_handle_time_mins', 'reopen_rate'], cmap='Reds'),
            use_container_width=True
        )

    with c2:
        st.subheader("Actionable Insights")
        worst_tech = df_tech.sort_values(by='reopen_rate', ascending=False).iloc[0]
        st.error(f"⚠️ **Attention Needed:** {worst_tech['full_name']}")
        st.write(f"Technician has a **{worst_tech['reopen_rate']}% Re-open Rate**.")