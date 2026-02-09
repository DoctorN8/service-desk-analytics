import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Service Desk Analytics", layout="wide")

# --- DATA CONNECTION ---
@st.cache_resource
def get_connection():
    conn = duckdb.connect('service_desk.duckdb', read_only=True)
    return conn

con = get_connection()

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Executive Pulse (VP View)", 
    "Technician Bottlenecks (Ops View)",
    "📈 Forecast Dashboard"
])
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

# =========================================================
# PAGE 3: FORECAST DASHBOARD
# =========================================================
elif page == "📈 Forecast Dashboard":
    st.title("📈 Ticket Volume & Backlog Forecasting")
    st.markdown("SARIMAX and Prophet models predicting 4-8 week service desk trends.")
    
    # Forecast horizon selector
    forecast_weeks = st.slider("Forecast Horizon (weeks)", min_value=4, max_value=8, value=6)
    forecast_days = forecast_weeks * 7
    
    st.divider()
    
    # --- TICKET VOLUME FORECAST ---
    st.subheader("🎫 Daily Ticket Volume Forecast")
    
    @st.cache_data(ttl=3600)
    def get_ticket_volume_data():
        # Aggregate daily ticket counts from your tickets table
        query = """
        SELECT 
            DATE_TRUNC('day', created_at) as date,
            COUNT(*) as tickets_created
        FROM tickets
        GROUP BY DATE_TRUNC('day', created_at)
        ORDER BY date
        """
        return con.sql(query).df()
    
    @st.cache_data(ttl=3600)
    def forecast_ticket_volume(forecast_days):
        df = get_ticket_volume_data()
        
        # Prepare data for Prophet
        df_prophet = df.rename(columns={'date': 'ds', 'tickets_created': 'y'})
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df_prophet)
        
        # Generate future dates
        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast = model.predict(future)
        
        return df_prophet, forecast
    
    try:
        df_volume, forecast_volume = forecast_ticket_volume(forecast_days)
        
        # Plot historical + forecast
        fig_volume = go.Figure()
        
        # Historical data
        fig_volume.add_trace(go.Scatter(
            x=df_volume['ds'],
            y=df_volume['y'],
            mode='lines',
            name='Historical',
            line=dict(color='#1f77b4')
        ))
        
        # Forecast
        fig_volume.add_trace(go.Scatter(
            x=forecast_volume['ds'],
            y=forecast_volume['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        # Confidence interval
        fig_volume.add_trace(go.Scatter(
            x=forecast_volume['ds'],
            y=forecast_volume['yhat_upper'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(255, 127, 14, 0)'),
            showlegend=False
        ))
        
        fig_volume.add_trace(go.Scatter(
            x=forecast_volume['ds'],
            y=forecast_volume['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255, 127, 14, 0)'),
            name='95% Confidence',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ))
        
        fig_volume.update_layout(
            title="Daily Ticket Volume: Historical vs. Forecast",
            xaxis_title="Date",
            yaxis_title="Tickets Created",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        future_forecast = forecast_volume.tail(forecast_days)
        
        with col1:
            avg_forecast = future_forecast['yhat'].mean()
            st.metric("Avg Daily Tickets (Forecast)", f"{avg_forecast:.0f}")
        
        with col2:
            peak_forecast = future_forecast['yhat'].max()
            st.metric("Peak Day (Forecast)", f"{peak_forecast:.0f}")
        
        with col3:
            total_forecast = future_forecast['yhat'].sum()
            st.metric(f"Total Volume ({forecast_weeks} weeks)", f"{total_forecast:.0f}")
            
    except Exception as e:
        st.error(f"Error generating ticket volume forecast: {e}")
    
    st.divider()
    
    # --- BACKLOG FORECAST ---
    st.subheader("📦 Backlog Level Forecast")
    
    @st.cache_data(ttl=3600)
    def get_backlog_data():
        query = """
        SELECT * 
        FROM vw_kpi_backlog_history 
        ORDER BY full_date
        """
        return con.sql(query).df()
    
    @st.cache_data(ttl=3600)
    def forecast_backlog(forecast_days):
        df = get_backlog_data()
        
        # Prepare for Prophet
        df_prophet = df[['full_date', 'total_backlog']].rename(
            columns={'full_date': 'ds', 'total_backlog': 'y'}
        )
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=False,
            changepoint_prior_scale=0.1
        )
        model.fit(df_prophet)
        
        # Generate future dates
        future = model.make_future_dataframe(periods=forecast_days, freq='D')
        forecast = model.predict(future)
        
        return df_prophet, forecast
    
    try:
        df_backlog, forecast_backlog = forecast_backlog(forecast_days)
        
        # Plot
        fig_backlog = go.Figure()
        
        # Historical
        fig_backlog.add_trace(go.Scatter(
            x=df_backlog['ds'],
            y=df_backlog['y'],
            mode='lines',
            name='Historical Backlog',
            line=dict(color='#d62728')
        ))
        
        # Forecast
        fig_backlog.add_trace(go.Scatter(
            x=forecast_backlog['ds'],
            y=forecast_backlog['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', dash='dash')
        ))
        
        # Confidence interval
        fig_backlog.add_trace(go.Scatter(
            x=forecast_backlog['ds'],
            y=forecast_backlog['yhat_upper'],
            fill=None,
            mode='lines',
            line=dict(color='rgba(255, 127, 14, 0)'),
            showlegend=False
        ))
        
        fig_backlog.add_trace(go.Scatter(
            x=forecast_backlog['ds'],
            y=forecast_backlog['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line=dict(color='rgba(255, 127, 14, 0)'),
            name='95% Confidence',
            fillcolor='rgba(255, 127, 14, 0.2)'
        ))
        
        fig_backlog.update_layout(
            title="Backlog Growth: Historical vs. Forecast",
            xaxis_title="Date",
            yaxis_title="Active Backlog Size",
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig_backlog, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        future_backlog = forecast_backlog.tail(forecast_days)
        
        with col1:
            avg_backlog = future_backlog['yhat'].mean()
            st.metric("Avg Backlog (Forecast)", f"{avg_backlog:.0f}")
        
        with col2:
            peak_backlog = future_backlog['yhat'].max()
            st.metric("Peak Backlog (Forecast)", f"{peak_backlog:.0f}")
        
        with col3:
            current_backlog = df_backlog['y'].iloc[-1]
            backlog_change = peak_backlog - current_backlog
            st.metric("Projected Growth", f"{backlog_change:+.0f}", 
                     delta_color="inverse")
            
    except Exception as e:
        st.error(f"Error generating backlog forecast: {e}")
    
    st.divider()
    
    # Model info
    with st.expander("ℹ️ About the Forecasting Models"):
        st.markdown("""
        **Prophet** (Facebook's forecasting library) is used for both forecasts:
        
        - **Ticket Volume**: Captures daily and weekly seasonality patterns
        - **Backlog**: Focuses on weekly trends with trend changepoint detection
        
        The models are trained on historical data from your DuckDB warehouse and provide:
        - Point forecasts (orange dashed line)
        - 95%
