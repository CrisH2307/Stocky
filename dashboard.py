import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd
import numpy as np
import logging

# --- Page Configuration ---
st.set_page_config(
    page_title="Volatility Forecast Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
def create_metric_card(title, value, suffix=""):
    st.markdown(f'<div class="metric-card"><h3>{title}</h3><p>{value}{suffix}</p></div>', unsafe_allow_html=True)

# --- Backend Integration Functions ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_sp500_analysis():
    """Returns a full analysis of the S&P 500 data from the local Excel file."""
    try:
        from local_data_processor import load_excel_data, analyze_sp500_data
        df = load_excel_data('data/data_SP500.xlsx', 'S&P 500')
        if df is None:
            return {"error": "Could not load S&P 500 data file."}
        return analyze_sp500_data(df)
    except Exception as e:
        logger.error(f"Error in S&P 500 analysis: {e}")
        return {"error": f"Error analyzing S&P 500 data: {str(e)}"}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_vix_analysis():
    """Returns a full analysis of the VIX data from the local Excel file."""
    try:
        from local_data_processor import load_excel_data, analyze_vix_data
        df = load_excel_data('data/data_VIX.xlsx', 'VIX')
        if df is None:
            return {"error": "Could not load VIX data file."}
        return analyze_vix_data(df)
    except Exception as e:
        logger.error(f"Error in VIX analysis: {e}")
        return {"error": f"Error analyzing VIX data: {str(e)}"}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_ticker_analysis(ticker_symbol: str):
    """Analyzes a given stock ticker using live data from stooq."""
    try:
        from ticker_analyzer import analyze_ticker
        analysis_result = analyze_ticker(ticker_symbol)
        if analysis_result is None:
            return {"error": f"Could not retrieve data for ticker '{ticker_symbol}'. Is it a valid symbol on stooq?"}
        return analysis_result
    except Exception as e:
        logger.error(f"Error in ticker analysis: {e}")
        return {"error": f"Error analyzing ticker {ticker_symbol}: {str(e)}"}

@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_lstm_garch_performance_data():
    """Returns the train/test predictions and actuals from the LSTM-GARCH model."""
    try:
        from lstm_garch_model import get_lstm_garch_performance_plot
        result = get_lstm_garch_performance_plot()
        if result is None:
            return {"error": "Failed to run LSTM-GARCH model pipeline."}
        return result
    except Exception as e:
        logger.error(f"Error in LSTM-GARCH performance: {e}")
        return {"error": f"Error running LSTM-GARCH model: {str(e)}"}

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_7_day_forecast():
    """Returns a 7-day ahead volatility forecast from the LSTM-GARCH model."""
    try:
        from lstm_garch_model import get_7_day_lstm_garch_forecast
        result = get_7_day_lstm_garch_forecast()
        if result is None:
            return {"error": "Failed to generate 7-day forecast."}
        return result
    except Exception as e:
        logger.error(f"Error in 7-day forecast: {e}")
        return {"error": f"Error generating 7-day forecast: {str(e)}"}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_garch_prediction(forecast_horizon: int = 30):
    """Runs the GARCH(2,2) rolling forecast."""
    try:
        from garch_model import load_and_prepare_data, run_garch_forecast
        returns = load_and_prepare_data('data/data_SP500.xlsx')
        if returns is None:
            return {"error": "Failed to load data for GARCH model."}
        
        predictions, dates = run_garch_forecast(returns, forecast_horizon=forecast_horizon)
        if not predictions:
            return {"error": "GARCH model failed to generate a forecast."}
        
        return {"dates": dates, "predicted_volatility": predictions}
    except Exception as e:
        logger.error(f"Error in GARCH prediction: {e}")
        return {"error": f"Error running GARCH model: {str(e)}"}

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def analyze_portfolio_data(portfolio_string: str):
    """Analyzes a portfolio of stocks given a string of tickers and weights."""
    try:
        from portfolio_analyzer import analyze_portfolio
        analysis_result = analyze_portfolio(portfolio_string)
        if "error" in analysis_result:
            return {"error": analysis_result["error"]}
        return analysis_result
    except Exception as e:
        logger.error(f"Error in portfolio analysis: {e}")
        return {"error": f"Error analyzing portfolio: {str(e)}"}

def calculate_live_error_metrics(historical_vol, train_vol, test_vol):
    """
    Calculate live error metrics for stock analysis using the same approach as the notebook.
    Returns MAE and RMSE for both train and test sets.
    """
    train_mae = 0.0
    train_rmse = 0.0
    test_mae = 0.0
    test_rmse = 0.0
    
    try:
        # Calculate train metrics - compare predicted vs actual (like y_train_inv vs train_pred_inv)
        if train_vol is not None and len(train_vol) > 0:
            train_actual = historical_vol.loc[train_vol.index]
            # Remove any NaN values
            valid_mask = ~(np.isnan(train_vol) | np.isnan(train_actual))
            if np.sum(valid_mask) > 0:
                # Convert to numpy arrays for calculation
                y_train_inv = train_actual[valid_mask].values
                train_pred_inv = train_vol[valid_mask].values
                train_mae = np.mean(np.abs(y_train_inv - train_pred_inv))
                train_rmse = np.sqrt(np.mean((y_train_inv - train_pred_inv)**2))
        
        # Calculate test metrics - compare predicted vs actual (like y_test_inv vs test_pred_inv)
        if test_vol is not None and len(test_vol) > 0:
            test_actual = historical_vol.loc[test_vol.index]
            # Remove any NaN values
            valid_mask = ~(np.isnan(test_vol) | np.isnan(test_actual))
            if np.sum(valid_mask) > 0:
                # Convert to numpy arrays for calculation
                y_test_inv = test_actual[valid_mask].values
                test_pred_inv = test_vol[valid_mask].values
                test_mae = np.mean(np.abs(y_test_inv - test_pred_inv))
                test_rmse = np.sqrt(np.mean((y_test_inv - test_pred_inv)**2))
                
    except Exception as e:
        logger.warning(f"Error calculating live metrics: {e}")
    
    return train_mae, train_rmse, test_mae, test_rmse

def set_garch_alert(threshold: float, phone_number: str):
    """Sets a volatility alert (simplified version for Streamlit)."""
    try:
        # For Streamlit, we'll just store the alert in session state
        # In a real implementation, you might want to use a database or external service
        st.session_state['garch_alert'] = {
            "phone_number": phone_number,
            "threshold": threshold
        }
        return {"message": "GARCH volatility alert has been set successfully."}
    except Exception as e:
        logger.error(f"Error setting GARCH alert: {e}")
        return {"error": f"Error setting alert: {str(e)}"}

# --- Plotting Functions ---
def plot_sp500_returns(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['dates'], y=chart_data['returns'], mode='lines', name='Market Returns', line=dict(color='#1f77b4', width=2), fill='tozeroy', fillcolor='rgba(31, 119, 180, 0.1)'))
    fig.update_layout(title="Market from 2000 - Now", yaxis_title="Returns (%)", template="plotly_dark", height=400, showlegend=False, margin=dict(l=10, r=10, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_7_day_forecast(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['dates'], y=chart_data['forecast'], mode='lines+markers', name='7-Day Forecast', line=dict(color='#00ff88', width=3), marker=dict(size=8)))
    fig.update_layout(title="Market 7-Day Volatility Forecast (LSTM-GARCH)", yaxis_title="Predicted Annualized Volatility (%)", template="plotly_dark", height=500, margin=dict(l=10, r=10, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_model_performance(chart_data: dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['train_dates'] + chart_data['test_dates'], y=chart_data['train_actual'] + chart_data['test_actual'], mode='lines', name='Actual Volatility', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=chart_data['train_dates'], y=chart_data['train_pred'], mode='lines', name='Train Predict (LSTM-GARCH)', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=chart_data['test_dates'], y=chart_data['test_pred'], mode='lines', name='Test Predict (LSTM-GARCH)', line=dict(color='blue', dash='dot')))
    fig.update_layout(title="Market LSTM-GARCH Model Performance", yaxis_title="Annualized Volatility (%)", xaxis_title="Date", template="plotly_dark", height=500, showlegend=True, legend=dict(x=0.01, y=0.99), margin=dict(l=10, r=10, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_stock_returns(chart_data: dict):
    # Check if data has error or missing required keys
    if chart_data.get("error") or 'returns_dates' not in chart_data or 'returns_values' not in chart_data:
        st.error("No valid returns data available for plotting.")
        return
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['returns_dates'], y=chart_data['returns_values'], mode='lines', name='Returns', line=dict(color='#1f77b4')))
    fig.update_layout(title=f"Daily Returns for {chart_data['ticker']}", yaxis_title="Daily Returns (%)", template="plotly_dark", height=500, margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_stock_volatility(chart_data: dict):
    # Check if data has error or missing required keys
    if chart_data.get("error") or 'historical_vol_dates' not in chart_data or 'forecast_dates' not in chart_data:
        st.error("No valid volatility data available for plotting.")
        return
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['historical_vol_dates'], y=chart_data['historical_vol_values'], mode='lines', name='Historical Volatility', line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=chart_data['forecast_dates'], y=chart_data['forecast_values'], mode='lines+markers', name='Forecasted Volatility', line=dict(color='blue', dash='dot'), marker=dict(size=6)))
    fig.update_layout(title=f"Historical Volatility vs Predicted Volatility for {chart_data['ticker']}", yaxis_title="Annualized Volatility (%)", template="plotly_dark", height=500, showlegend=True, legend=dict(x=0.01, y=0.99), margin=dict(l=20, r=20, t=60, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_stock_forecast_only(chart_data: dict):
    # Check if data has error or missing required keys
    if chart_data.get("error") or 'forecast_dates' not in chart_data or 'forecast_values' not in chart_data:
        st.error("No valid forecast data available for plotting.")
        return
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['forecast_dates'],
        y=chart_data['forecast_values'],
        mode='lines+markers', name='Forecasted Volatility',
        line=dict(color='#2ca02c'), marker=dict(size=8)
    ))
    fig.update_layout(
        title=f"7-Day Annualized Implied Volatility for {chart_data['ticker']}",
        yaxis_title="Predicted Annualized Volatility (%)",
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
        yaxis_title="Annualized Volatility (%)",
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
    # Check if data has error or missing required keys
    if chart_data.get("error"):
        st.error(f"Error in data: {chart_data['error']}")
        return
        
    fig = go.Figure()
    
    # Line 1: Historical Volatility
    if chart_data.get('historical_vol_dates') and chart_data.get('historical_vol_values'):
        fig.add_trace(go.Scatter(
            x=chart_data.get('historical_vol_dates'), 
            y=chart_data.get('historical_vol_values'),
            mode='lines', name='Historical Volatility',
            line=dict(color='rgba(255, 255, 255, 0.6)', dash='dot')
        ))
    
    # Line 2: In-Sample (Training) Predicted Volatility
    if chart_data.get('train_vol_dates') and chart_data.get('train_vol_values'):
        fig.add_trace(go.Scatter(
            x=chart_data.get('train_vol_dates'), 
            y=chart_data.get('train_vol_values'),
            mode='lines', name='Train Predict (LSTM-GARCH)',
            line=dict(color='#00CC96') # Green
        ))
    
    # Line 3: Out-of-Sample (Testing) Predicted Volatility
    if chart_data.get('test_vol_dates') and chart_data.get('test_vol_values'):
        fig.add_trace(go.Scatter(
            x=chart_data.get('test_vol_dates'), 
            y=chart_data.get('test_vol_values'),
            mode='lines', name='Test Predict (LSTM-GARCH)',
            line=dict(color='#636EFA') # Blue
        ))

    # Check if any traces were added
    if not fig.data:
        st.error("No valid volatility data available for plotting.")
        return

    fig.update_layout(
        title=title,
        yaxis_title="Annualized Volatility (%)",
        template="plotly_dark", height=500,
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_forecast_chart(chart_data, title, y_axis_title="Predicted Annualized Volatility (%)"):
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
        yaxis_title="Predicted Annualized Volatility (%)",
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
        yaxis_title="Annualized Volatility (%)",
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
            st.session_state.market_volatility_data = get_lstm_garch_performance_data()
            st.session_state.market_returns_data = get_sp500_analysis()
            st.session_state.market_forecast_data = get_7_day_forecast()
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
            
            # Use error metrics from the model if available, otherwise calculate locally
            if 'test_mae' in data and 'test_rmse' in data and 'train_mae' in data and 'train_rmse' in data:
                test_mae = data['test_mae']
                test_rmse = data['test_rmse']
                train_mae = data['train_mae']
                train_rmse = data['train_rmse']
            else:
                # Fallback calculation
                y_true = np.array(data['test_actual'])
                y_pred = np.array(data['test_pred'])
                test_mae = np.mean(np.abs(y_true - y_pred)) * 100
                test_rmse = np.sqrt(np.mean((y_true - y_pred)**2)) * 100
                train_mae = 0.0  # Placeholder if not available
                train_rmse = 0.0  # Placeholder if not available
            
            # Train & Test Split box with same structure and height
            split_metrics_html = f"""
            <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; height: 180px; padding: 25px;">
                <h3>Train & Test Split</h3>
                <p style="font-size: 2.2rem; font-weight: bold; margin: 0;">80/20</p>
            </div>
            """
            with col1: st.markdown(split_metrics_html, unsafe_allow_html=True)
            
            # Test Set Metrics in one box
            test_metrics_html = f"""
            <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; height: 180px; padding: 25px;">
                <h3>Test Set Performance</h3>
                <p style="font-size: 1.8rem; margin-bottom: 10px;">MAE: {test_mae:.4f}%</p>
                <p style="font-size: 1.8rem; margin: 0;">RMSE: {test_rmse:.4f}%</p>
            </div>
            """
            with col2: st.markdown(test_metrics_html, unsafe_allow_html=True)
            
            # Train Set Metrics in one box
            train_metrics_html = f"""
            <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; height: 180px; padding: 25px;">
                <h3>Train Set Performance</h3>
                <p style="font-size: 1.8rem; margin-bottom: 10px;">MAE: {train_mae:.4f}%</p>
                <p style="font-size: 1.8rem; margin: 0;">RMSE: {train_rmse:.4f}%</p>
            </div>
            """
            with col3: st.markdown(train_metrics_html, unsafe_allow_html=True)
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
                'Predicted Annualized Volatility (%)': [f"{x:.6f}" for x in forecast_data['forecast']]
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
            st.session_state['stock_analysis_data'] = get_ticker_analysis(ticker_symbol)
            st.session_state['last_analyzed_stock'] = ticker_symbol
    
    if 'stock_analysis_data' in st.session_state and st.session_state.get('stock_analysis_data'):
        data = st.session_state['stock_analysis_data']
        
        # Check if there's an error in the data
        if data.get("error"):
            st.error(f"Error analyzing {st.session_state['last_analyzed_stock']}: {data['error']}")
            st.info("💡 **Tip**: Make sure you're using a valid stock ticker symbol (e.g., 'AAPL' for Apple, 'GOOGL' for Google, 'MSFT' for Microsoft)")
        else:
            st.markdown(f"---")
            st.markdown(f"### Showing analysis for **{st.session_state['last_analyzed_stock']}**")
            view_option = st.selectbox("Select View:", ["Returns", "Volatility", "Implied Volatility"], key='stock_view')

            if view_option == "Returns":
                plot_stock_returns(data)
                st.markdown("### Key Return Metrics")
                col1, col2, col3 = st.columns(3)
                returns_values = data.get('returns_values', [])
                if returns_values:
                    avg_return = np.mean(returns_values)
                    max_return = max(returns_values)
                    min_return = min(returns_values)
                    with col1: create_metric_card("Average Daily Return", f"{avg_return:.3f}", suffix="%")
                    with col2: create_metric_card("Max Daily Return", f"{max_return:.2f}", suffix="%")
                    with col3: create_metric_card("Min Daily Return", f"{min_return:.2f}", suffix="%")
                else:
                    st.warning("No returns data available for metrics.")
            
            elif view_option == "Volatility":
                plot_volatility_chart(data, f"Historical vs. Predicted Volatility for {st.session_state['last_analyzed_stock']}")
                
                # Add error metrics for stock analysis
                st.markdown("### Error Performance Metrics")
                col1, col2, col3 = st.columns(3)
                
                # Calculate live error metrics for the selected stock
                try:
                    # Get the volatility data for live calculation
                    historical_vol_values = data.get('historical_vol_values', [])
                    train_vol_values = data.get('train_vol_values', [])
                    test_vol_values = data.get('test_vol_values', [])
                    
                    if historical_vol_values and train_vol_values and test_vol_values:
                        # Convert to pandas Series for calculation
                        historical_vol = pd.Series(historical_vol_values, index=pd.to_datetime(data.get('historical_vol_dates', [])))
                        train_vol = pd.Series(train_vol_values, index=pd.to_datetime(data.get('train_vol_dates', [])))
                        test_vol = pd.Series(test_vol_values, index=pd.to_datetime(data.get('test_vol_dates', [])))
                        
                        # Calculate live metrics
                        train_mae, train_rmse, test_mae, test_rmse = calculate_live_error_metrics(historical_vol, train_vol, test_vol)
                        
                        st.info(f"Live calculated metrics for {st.session_state['last_analyzed_stock']}: Train MAE: {train_mae:.4f}%, Test MAE: {test_mae:.4f}%")
                    else:
                        # Fallback to stored metrics if available
                        if 'test_mae' in data and 'test_rmse' in data and 'train_mae' in data and 'train_rmse' in data:
                            test_mae = data['test_mae']
                            test_rmse = data['test_rmse']
                            train_mae = data['train_mae']
                            train_rmse = data['train_rmse']
                        else:
                            # Default values
                            test_mae = 0.0
                            test_rmse = 0.0
                            train_mae = 0.0
                            train_rmse = 0.0
                except Exception as e:
                    logger.warning(f"Error in live metrics calculation: {e}")
                    # Fallback values
                    test_mae = 0.0
                    test_rmse = 0.0
                    train_mae = 0.0
                    train_rmse = 0.0
                
                # Train & Test Split box with same structure and height
                split_metrics_html = f"""
                <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; height: 180px; padding: 25px;">
                    <h3>Train & Test Split</h3>
                    <p style="font-size: 2.2rem; font-weight: bold; margin: 0;">80/20</p>
                </div>
                """
                with col1: st.markdown(split_metrics_html, unsafe_allow_html=True)
                
                # Test Set Metrics in one box
                test_metrics_html = f"""
                <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; height: 180px; padding: 25px;">
                    <h3>Test Set Performance</h3>
                    <p style="font-size: 1.8rem; margin-bottom: 10px;">MAE: {test_mae:.4f}%</p>
                    <p style="font-size: 1.8rem; margin: 0;">RMSE: {test_rmse:.4f}%</p>
                </div>
                """
                with col2: st.markdown(test_metrics_html, unsafe_allow_html=True)
                
                # Train Set Metrics in one box
                train_metrics_html = f"""
                <div class="metric-card" style="display: flex; flex-direction: column; justify-content: center; height: 180px; padding: 25px;">
                    <h3>Train Set Performance</h3>
                    <p style="font-size: 1.8rem; margin-bottom: 10px;">MAE: {train_mae:.4f}%</p>
                    <p style="font-size: 1.8rem; margin: 0;">RMSE: {train_rmse:.4f}%</p>
                </div>
                """
                with col3: st.markdown(train_metrics_html, unsafe_allow_html=True)

            elif view_option == "Implied Volatility":
                plot_stock_forecast_only(data)
                if 'forecast_dates' in data and 'forecast_values' in data:
                    df_forecast = pd.DataFrame({
                        'Date': data['forecast_dates'],
                        'Predicted Annualized Volatility (%)': [f"{x:.6f}" for x in data['forecast_values']]
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
            st.session_state['portfolio_data'] = analyze_portfolio_data(portfolio_string)

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
            result = set_garch_alert(threshold, phone_number)
            if "error" not in result:
                st.sidebar.success(f"Alert set for {threshold}% threshold")
            else:
                st.sidebar.error(f"Failed to set alert: {result['error']}")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #777;'>Volatility Forecast Dashboard | Professional Financial Analytics</div>", unsafe_allow_html=True)