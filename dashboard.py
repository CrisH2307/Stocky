import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(
    page_title="Volatility Forecast Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Configuration ---
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- UI Styling ---
st.markdown("""
    <style>
    .stApp { background-color: #111111; color: #EAEAEA; }
    .metric-card { background-color: #222222; border-radius: 10px; padding: 20px; text-align: center; border: 1px solid #444444; margin-bottom: 20px; }
    .metric-card h3 { font-size: 1.2rem; color: #A0A0A0; margin-bottom: 5px; }
    .metric-card p { font-size: 2.2rem; font-weight: bold; margin: 0; }
    .section-header { border-bottom: 2px solid #444; padding-bottom: 10px; margin-top: 20px; margin-bottom: 20px; font-size: 2rem; font-weight: bold; color: #FAFAFA; }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def make_api_request(endpoint: str, method: str = 'GET', json_payload: dict = None):
    try:
        if method.upper() == 'POST':
            response = requests.post(f"{API_BASE_URL}{endpoint}", json=json_payload)
        else:
            response = requests.get(f"{API_BASE_URL}{endpoint}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        detail = "API Error. Is the backend running?"
        try:
            detail = e.response.json().get('detail', detail)
        except (AttributeError, ValueError):
            pass
        st.error(f"{detail}")
        return None

def create_metric_card(title, value, suffix=""):
    st.markdown(f'<div class="metric-card"><h3>{title}</h3><p>{value}{suffix}</p></div>', unsafe_allow_html=True)

# --- Plotting Functions ---
def plot_sp500_returns(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['dates'], y=chart_data['returns'], mode='lines', name='Market Returns', line=dict(color='#1f77b4', width=2), fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)'))
    fig.update_layout(title="Market from 2000 - Now", yaxis_title="Returns (%)", template="plotly_dark", height=400, showlegend=False, margin=dict(l=10, r=10, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_7_day_forecast(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['dates'], y=chart_data['forecast'], mode='lines+markers', name='7-Day Forecast', line=dict(color='#00ff88', width=3), marker=dict(size=8)))
    fig.update_layout(title="Market 7-Day Volatility Forecast (LSTM-GARCH)", yaxis_title="Predicted Daily Volatility (%)", template="plotly_dark", height=500, margin=dict(l=10, r=10, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_model_performance(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['train_dates'] + chart_data['test_dates'], y=chart_data['train_actual'] + chart_data['test_actual'], mode='lines', name='Actual Volatility', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=chart_data['train_dates'], y=chart_data['train_pred'], mode='lines', name='Train Predict (LSTM-GARCH)', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=chart_data['test_dates'], y=chart_data['test_pred'], mode='lines', name='Test Predict (LSTM-GARCH)', line=dict(color='blue', dash='dot')))
    fig.update_layout(title="Market LSTM-GARCH Model Performance", yaxis_title="Daily Volatility (%)", xaxis_title="Date", template="plotly_dark", height=500, showlegend=True, legend=dict(x=0.01, y=0.99), margin=dict(l=10, r=10, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_stock_returns(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['returns_dates'], y=chart_data['returns_values'], mode='lines', name='Returns', line=dict(color='#1f77b4')))
    fig.update_layout(title=f"Daily Returns for {chart_data['ticker']}", yaxis_title="Daily Returns (%)", template="plotly_dark", height=500, margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_stock_volatility(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['historical_vol_dates'], y=chart_data['historical_vol_values'], mode='lines', name='Historical Volatility', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=chart_data['forecast_dates'], y=chart_data['forecast_values'], mode='lines+markers', name='Forecasted Volatility', line=dict(color='blue', dash='dot'), marker=dict(size=6)))
    fig.update_layout(title=f"Historical Volatility vs Predicted Volatility for {chart_data['ticker']}", yaxis_title="Daily Volatility (%)", template="plotly_dark", height=500, showlegend=True, legend=dict(x=0.01, y=0.99), margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_stock_forecast_only(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['forecast_dates'],
        y=chart_data['forecast_values'],
        mode='lines+markers', name='Forecasted Volatility',
        line=dict(color='#2ca02c'), marker=dict(size=8)
    ))
    fig.update_layout(
        title=f"7-Day Daily Implied Volatility for {chart_data['ticker']}",
        yaxis_title="Predicted Daily Volatility (%)",
        template="plotly_dark", height=500, showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_portfolio_volatility(chart_data: dict):
    """Plots the volatility of a portfolio and its individual assets."""
    fig = go.Figure()
    
    # Define a color sequence
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
    
    # Plot individual asset volatilities
    individual_vols = chart_data.get('individual_volatilities', {})
    for i, (ticker, data) in enumerate(individual_vols.items()):
        fig.add_trace(go.Scatter(
            x=data['dates'],
            y=data['volatility'],
            mode='lines',
            name=f'{ticker} Volatility',
            line=dict(color=colors[i % len(colors)], dash='dot', width=1.5),
            opacity=0.8
        ))

    # Plot the combined portfolio volatility
    portfolio_vol = chart_data.get('portfolio_volatility', {})
    if portfolio_vol:
        fig.add_trace(go.Scatter(
            x=portfolio_vol['dates'],
            y=portfolio_vol['volatility'],
            mode='lines',
            name='Total Portfolio Volatility',
            line=dict(color='white', width=3)
        ))

    fig.update_layout(
        title="Portfolio and Asset Volatility",
        yaxis_title="Daily Volatility (%)",
        template="plotly_dark",
        height=500,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_volatility_chart(chart_data, title):
    fig = go.Figure()
    
    # Line 1: Historical Volatility
    fig.add_trace(go.Scatter(
        x=chart_data.get('historical_vol_dates'), 
        y=chart_data.get('historical_vol_values'),
        mode='lines', name='Historical Volatility',
        line=dict(color='rgba(255, 255, 255, 0.6)', dash='dot')
    ))
    
    # Line 2: In-Sample (Training) Predicted Volatility
    fig.add_trace(go.Scatter(
        x=chart_data.get('train_vol_dates'), 
        y=chart_data.get('train_vol_values'),
        mode='lines', name='Train Predict (LSTM-GARCH)',
        line=dict(color='#00CC96') # Green
    ))
    
    # Line 3: Out-of-Sample (Testing) Predicted Volatility
    fig.add_trace(go.Scatter(
        x=chart_data.get('test_vol_dates'), 
        y=chart_data.get('test_vol_values'),
        mode='lines', name='Test Predict (LSTM-GARCH)',
        line=dict(color='#EF553B') # Red
    ))

    fig.update_layout(
        title=title,
        yaxis_title="Daily Volatility (%)",
        template="plotly_dark", height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_forecast_chart(chart_data, title, y_axis_title="Predicted Daily Volatility (%)"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['dates'], 
        y=chart_data['volatility'], 
        mode='lines+markers',
        name='7-Day Forecast'
    ))
    fig.update_layout(
        title=title,
        yaxis_title=y_axis_title,
        template="plotly_dark",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_portfolio_forecast(chart_data: dict):
    """Plots the 7-day volatility forecast for a portfolio and its assets."""
    fig = go.Figure()
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692']
    
    individual_forecasts = chart_data.get('individual_forecasts', {})
    if individual_forecasts:
        for i, (ticker, data) in enumerate(individual_forecasts.items()):
            fig.add_trace(go.Scatter(
                x=data.get('dates'), 
                y=data.get('volatility'), 
                mode='lines',
                name=f'{ticker} Forecast', 
                line=dict(color=colors[i % len(colors)], dash='dot')
            ))

    portfolio_forecast = chart_data.get('portfolio_forecast', {})
    if portfolio_forecast and portfolio_forecast.get('dates'):
        fig.add_trace(go.Scatter(
            x=portfolio_forecast.get('dates'), 
            y=portfolio_forecast.get('volatility'),
            mode='lines+markers', name='Total Portfolio Forecast',
            line=dict(color='white', width=3), marker=dict(size=8)
        ))

    fig.update_layout(
        yaxis_title="Predicted Daily Volatility (%)",
        template="plotly_dark", height=500, showlegend=True,
        legend=dict(x=0.01, y=0.99), margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_market_volatility(chart_data):
    """Plots the market volatility (actual vs. predicted)."""
    fig = go.Figure()
    
    # Actual Volatility
    fig.add_trace(go.Scatter(
        x=chart_data.get('dates'), 
        y=chart_data.get('actual_vol'),
        mode='lines', name='Actual Volatility',
        line=dict(color='#EF553B') # Red, solid
    ))
    
    # Train Predict
    fig.add_trace(go.Scatter(
        x=chart_data.get('train_dates'), 
        y=chart_data.get('train_pred'),
        mode='lines', name='Train Predict (LSTM-GARCH)',
        line=dict(color='orange', dash='dot')
    ))

    # Test Predict
    fig.add_trace(go.Scatter(
        x=chart_data.get('test_dates'), 
        y=chart_data.get('test_pred'),
        mode='lines', name='Test Predict (LSTM-GARCH)',
        line=dict(color='#636EFA', dash='dot') # Blue, dotted
    ))

    fig.update_layout(
        title="Market LSTM-GARCH Model Performance",
        yaxis_title="Daily Volatility (%)",
        template="plotly_dark", height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- Sidebar Navigation ---
st.sidebar.title("Analysis Dashboard")
st.sidebar.markdown("---")

# --- Main App ---
st.title("Analysis Dashboard")

analysis_mode = st.sidebar.selectbox(
    "Select Analysis Mode:",
    ["Market Analysis", "Stock Analysis", "Portfolio Analysis"],
    key='analysis_mode_selector'
)
st.sidebar.markdown("---")

# Based on selection, show different views
if analysis_mode == "Market Analysis":
    st.header("S&P 500 Market Analysis")

    # Automatically load data if it's not already in the session state
    if 'market_data_loaded' not in st.session_state:
        with st.spinner("Loading market analysis data... This is a one-time load."):
            st.session_state.market_volatility_data = make_api_request("/api/lstm-garch-performance")
            st.session_state.market_returns_data = make_api_request("/api/sp500")
            st.session_state.market_forecast_data = make_api_request("/api/lstm-garch-7-day-forecast")
            st.session_state.market_data_loaded = True

    view_option = st.selectbox("Select View:", ["Returns", "Volatility", "Implied Volatility"], key='market_view_selector')

    if view_option == "Returns":
        st.markdown("### Daily Returns")
        sp500_data = st.session_state.get('market_returns_data')
        if sp500_data and not sp500_data.get("error"):
            prices = sp500_data['chart_data']['prices']
            returns = [((prices[i] - prices[i-1]) / prices[i-1]) * 100 for i in range(1, len(prices))]
            returns_data = {'dates': sp500_data['chart_data']['dates'][1:], 'returns': returns}
            plot_sp500_returns(returns_data)
            st.markdown("### Key Return Metrics")
            col1, col2, col3 = st.columns(3)
            avg_return = np.mean(returns) if returns else 0
            max_return = max(returns) if returns else 0
            min_return = min(returns) if returns else 0
            with col1: create_metric_card("Average Daily Return", f"{avg_return:.3f}", suffix="%")
            with col2: create_metric_card("Max Daily Return", f"{max_return:.2f}", suffix="%")
            with col3: create_metric_card("Min Daily Return", f"{min_return:.2f}", suffix="%")
        else:
            st.warning("Could not load market returns data.")

    elif view_option == "Volatility":
        st.subheader("Market LSTM-GARCH Model Performance")
        data = st.session_state.get('market_volatility_data')
        if data and not data.get("error"):
            plot_model_performance(data)
            st.markdown("### Error Performance Metrics")
            col1, col2, col3 = st.columns(3)
            y_true = np.array(data['test_actual'])
            y_pred = np.array(data['test_pred'])
            test_mae = np.mean(np.abs(y_true - y_pred))
            test_rmse = np.sqrt(np.mean((y_true - y_pred)**2))
            with col1: create_metric_card("Train & Test Split", "80/20")
            with col2: create_metric_card("Mean Absolute Error (MAE)", f"{test_mae:.3f}")
            with col3: create_metric_card("Root Mean Squared Error (RMSE)", f"{test_rmse:.3f}")
        elif data and data.get("error"):
            st.error(data["error"])
        else:
            st.warning("Could not load market volatility data.")

    elif view_option == "Implied Volatility":
        st.subheader("7-Day Volatility Implied Volatility")
        forecast_data = st.session_state.get('market_forecast_data')
        if forecast_data and not forecast_data.get("error"):
            plot_7_day_forecast(forecast_data)
            st.markdown("### Forecast Values")
            df_forecast = pd.DataFrame({
                'Date': forecast_data['dates'],
                'Predicted Daily Volatility (%)': [f"{x:.6f}" for x in forecast_data['forecast']]
            })
            st.dataframe(df_forecast, use_container_width=True, hide_index=True)
        else:
            st.warning("Could not load market forecast data.")

elif analysis_mode == "Stock Analysis":
    st.header("Custom Stock Analysis")
    st.markdown("### Analyze any stock ticker using live data from `stooq`.")
    
    with st.form(key='stock_form'):
        ticker_symbol = st.text_input("Enter Stock Ticker:", "DIS").upper()
        analyze_button = st.form_submit_button(label='Analyze')

    if analyze_button and ticker_symbol:
        with st.spinner(f"Fetching data from stooq for {ticker_symbol}..."):
            st.session_state['stock_analysis_data'] = make_api_request(f"/api/ticker/{ticker_symbol}")
            st.session_state['last_analyzed_stock'] = ticker_symbol
    
    if 'stock_analysis_data' in st.session_state and st.session_state.get('stock_analysis_data'):
        st.markdown(f"---")
        st.markdown(f"### Showing analysis for **{st.session_state['last_analyzed_stock']}**")
        view_option = st.selectbox("Select View:", ["Returns", "Volatility", "Implied Volatility"], key='stock_view')
        data = st.session_state['stock_analysis_data']

        if view_option == "Returns":
            plot_stock_returns(data)
        
        elif view_option == "Volatility":
            plot_volatility_chart(data, f"Historical vs. Predicted Volatility for {ticker_symbol}")

        elif view_option == "Implied Volatility":
            plot_stock_forecast_only(data)
            df_forecast = pd.DataFrame({
                'Date': data['forecast_dates'],
                'Predicted Daily Volatility (%)': [f"{x:.6f}" for x in data['forecast_values']]
            })
            st.dataframe(df_forecast, use_container_width=True, hide_index=True)
    else:
        st.info("Enter a stock ticker and click 'Analyze'.")

elif analysis_mode == "Portfolio Analysis":
    st.title("Portfolio Performance Analysis")
    st.markdown("### Build your portfolio and analyze its historical performance.")

    if 'portfolio_items' not in st.session_state:
        st.session_state.portfolio_items = [
            {'ticker': 'AAPL', 'weight': 0.5},
            {'ticker': 'GOOG', 'weight': 0.3},
            {'ticker': 'DIS', 'weight': 0.2}
        ]

    st.markdown("#### Add Stock to Portfolio")
    with st.form("add_stock_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        ticker = col1.text_input("Ticker Symbol").upper()
        weight = col2.number_input("Weight", min_value=0.01, max_value=1.0, step=0.05, format="%.2f")
        
        if st.form_submit_button("Add Stock"):
            if ticker and not any(p['ticker'] == ticker for p in st.session_state.portfolio_items):
                st.session_state.portfolio_items.append({'ticker': ticker, 'weight': weight})
            elif ticker:
                st.warning(f"Ticker {ticker} is already in the portfolio.")

    st.markdown("---")
    st.markdown("#### Current Portfolio")
    
    total_weight = sum(p['weight'] for p in st.session_state.portfolio_items)
    
    c1, c2, c3 = st.columns([2, 1, 1])
    c1.write("**Ticker**")
    c2.write("**Weight**")
    
    for i, item in enumerate(st.session_state.portfolio_items):
        col1, col2, col3 = st.columns([2, 1, 1])
        col1.write(item['ticker'])
        col2.write(f"{item['weight']:.2f}")
        if col3.button("Remove", key=f"remove_{i}"):
            st.session_state.portfolio_items.pop(i)
            st.rerun()

    st.metric("Total Weight", f"{total_weight:.2f}")
    if not (0.99 <= total_weight <= 1.01):
        st.error("Portfolio weights must sum to 1.0 for a valid analysis.")
    
    # Simple button to trigger analysis from the list above
    if st.button("Analyze Portfolio", disabled=not (0.99 <= total_weight <= 1.01)):
        portfolio_string = ", ".join([f"{p['ticker']} {p['weight']}" for p in st.session_state.portfolio_items])
        
        # Clear old data before making a new request
        if 'portfolio_data' in st.session_state:
            del st.session_state['portfolio_data']
            
        with st.spinner("Analyzing portfolio forecast..."):
            response = requests.post(f"{API_BASE_URL}/analyze_portfolio/", json={"portfolio_string": portfolio_string})
            if response.status_code == 200:
                st.session_state['portfolio_data'] = response.json()
            else:
                st.error(f"Error from API: {response.text}")

    if 'portfolio_data' in st.session_state and st.session_state.get('portfolio_data'):
        data = st.session_state['portfolio_data']
        if "error" in data:
            st.error(data["error"])
            del st.session_state['portfolio_data']
        elif data.get('portfolio_forecast'):
            st.markdown("### Portfolio and Asset Implied Volatility")
            plot_portfolio_forecast(data)
            
            st.markdown("### Portfolio 7-Day Implied Volatility Data")
            portfolio_forecast_data = data.get('portfolio_forecast', {})
            if portfolio_forecast_data.get('dates'):
                df_vol = pd.DataFrame({
                    'Date': portfolio_forecast_data.get('dates', []),
                    'Predicted Portfolio Volatility (%)': [f"{v:.4f}" if v is not None else "N/A" for v in portfolio_forecast_data.get('volatility', [])]
                })
                st.dataframe(df_vol, use_container_width=True, height=300, hide_index=True)
    else:
        st.info("Build a portfolio with a total weight of 1.0, then click 'Analyze'.")

# --- Sidebar for Alerts ---
st.sidebar.title("Volatility Alerts")
st.sidebar.write("Get notified when the Market model predicts high volatility.")
with st.sidebar.form("alert_form"):
    phone_number = st.text_input("Phone Number", placeholder="+14155552671")
    threshold = st.number_input("Alert Threshold (%) for Market Forecast", min_value=0.1, value=1.5, step=0.1)
    submit_button = st.form_submit_button(label='Set Alert')
    if submit_button:
        if not phone_number or not threshold:
            st.sidebar.error("Please enter a valid phone number and threshold.")
        else:
            payload = {"phone_number": phone_number, "threshold": threshold}
            response = requests.post(f"{API_BASE_URL}/api/garch-alert", params=payload)
            if response and response.status_code == 200:
                st.sidebar.success(f"Alert set for {threshold}% threshold")
            else:
                error = (response.json() or {}).get('detail', 'Unknown error')
                st.sidebar.error(f"Failed to set alert: {error}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #777;'>Volatility Forecast Dashboard | Professional Financial Analytics</div>", unsafe_allow_html=True)