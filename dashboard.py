import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import pandas as pd

# --- Page Configuration ---
st.set_page_config(
    page_title="LSRM-GARCH Volatility Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- API Configuration ---
API_BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# --- UI Styling ---
st.markdown("""
    <style>
    /* Main app styling */
    .stApp {
        background-color: #111111;
        color: #EAEAEA;
    }
    /* Metric cards */
    .metric-card {
        background-color: #222222;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border: 1px solid #444444;
        margin-bottom: 20px;
    }
    .metric-card h3 {
        font-size: 1.2rem;
        color: #A0A0A0;
        margin-bottom: 5px;
    }
    .metric-card p {
        font-size: 2.2rem;
        font-weight: bold;
        margin: 0;
    }
    .metric-card .delta {
        font-size: 1rem;
        margin-top: 5px;
    }
    /* Section headers */
    .section-header {
        border-bottom: 2px solid #444;
        padding-bottom: 10px;
        margin-top: 20px;
        margin-bottom: 20px;
        font-size: 2rem;
        font-weight: bold;
        color: #FAFAFA;
    }
    /* Chart styling */
    .stPlotlyChart {
        border: 1px solid #333;
        border-radius: 8px;
    }
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a1a;
    }
    </style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def make_api_request(endpoint: str, params: dict = None):
    """Generic function to make API requests and handle errors."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch data from API: {e}")
        return None

def create_metric_card(title, value, delta=None, delta_color="normal", suffix=""):
    """Creates a styled metric card using HTML."""
    delta_html = ""
    if delta is not None:
        color = "green" if delta >= 0 else "red"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<p class="delta" style="color:{color};">{arrow} {delta:.2f}%</p>'
    
    st.markdown(f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <p>{value}{suffix}</p>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)

def plot_lsrm_garch_forecast(chart_data: dict):
    """Plots the LSTM-GARCH Volatility Forecast with enhanced styling."""
    fig = go.Figure()
    
    # Main forecast line
    fig.add_trace(go.Scatter(
        x=chart_data['dates'], 
        y=chart_data['predicted_volatility'],
        mode='lines+markers', 
        name='LSTM-GARCH Forecast', 
        line=dict(color='#00ff88', width=3),
        marker=dict(size=6, color='#00ff88')
    ))
    
    fig.update_layout(
        title={
            'text': "LSTM-GARCH Volatility Forecast Model (2000-Present)",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#FFFFFF'}
        },
        yaxis_title="Predicted Volatility (%)",
        xaxis_title="Forecast Date",
        template="plotly_dark",
        height=500,
        showlegend=True,
        margin=dict(l=10, r=10, t=80, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        yaxis=dict(
            gridcolor='#333333',
            zerolinecolor='#333333',
            title_font=dict(size=14)
        ),
        xaxis=dict(
            gridcolor='#333333',
            zerolinecolor='#333333',
            title_font=dict(size=14)
        )
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_spy_returns(chart_data: dict):
    """Plots S&P500 Returns over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['dates'], 
        y=chart_data['returns'],
        mode='lines', 
        name='S&P500 Returns', 
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy',
        fillcolor='rgba(31, 119, 180, 0.1)'
    ))
    fig.update_layout(
        title="S&P500 Returns Over Time (2000-Present)",
        yaxis_title="Returns (%)",
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_sp500_volatility(chart_data: dict):
    """Plots S&P500 Volatility over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['dates'], 
        y=chart_data['volatility'],
        mode='lines', 
        name='S&P500 Volatility', 
        line=dict(color='#ff7f0e', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 127, 14, 0.1)'
    ))
    fig.update_layout(
        title="S&P500 Volatility Over Time (2000-Present)",
        yaxis_title="Annualized Volatility (%)",
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

def plot_vix_overtime(chart_data: dict):
    """Plots VIX over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data['dates'], 
        y=chart_data['prices'],
        mode='lines', 
        name='VIX Index', 
        line=dict(color='#d62728', width=2),
        fill='tozeroy',
        fillcolor='rgba(214, 39, 40, 0.1)'
    ))
    fig.update_layout(
        title="VIX Index Over Time (2000-Present)",
        yaxis_title="VIX Level",
        template="plotly_dark",
        height=400,
        showlegend=False,
        margin=dict(l=10, r=10, t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# --- Sidebar Navigation ---
st.sidebar.title("📊 Market Analysis")
st.sidebar.markdown("---")

# Sidebar tabs for different analyses
analysis_type = st.sidebar.selectbox(
    "Select Analysis:",
    ["S&P500 Returns",  "S&P500 Volatility", "LSTM-GARCH Forecast"]
)

# --- Main App Layout ---
if analysis_type == "S&P500 Returns":
    st.title("📈 S&P500 Returns Analysis (2000-Present)")
    st.markdown("---")
    
    # Get SP500 data (we'll use this for S&P500 Returns)
    sp500_data = make_api_request("/api/sp500")
    if sp500_data:
        # Calculate returns from prices
        prices = sp500_data['chart_data']['prices']
        returns = []
        for i in range(1, len(prices)):
            ret = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
            returns.append(ret)
        
        # Create returns chart data
        returns_data = {
            'dates': sp500_data['chart_data']['dates'][1:],
            'returns': returns
        }
        
        plot_spy_returns(returns_data)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_return = sum(returns) / len(returns) if returns else 0
            create_metric_card("Average Daily Return", f"{avg_return:.3f}", suffix="%")
        with col2:
            max_return = max(returns) if returns else 0
            create_metric_card("Maximum Daily Return", f"{max_return:.2f}", suffix="%")
        with col3:
            min_return = min(returns) if returns else 0
            create_metric_card("Minimum Daily Return", f"{min_return:.2f}", suffix="%")
    else:
        st.error("Could not load SPY returns data.")

elif analysis_type == "LSTM-GARCH Forecast":
    # Main LSRM-GARCH Model Section
    st.title("🔮 LSTM-GARCH Volatility Forecast Model")
    st.markdown("### Advanced Machine Learning Model for S&P 500 Volatility Prediction")
    st.markdown("---")
    
    # Model description
    st.markdown("""
    <div style='background-color: #222222; padding: 20px; border-radius: 10px; border: 1px solid #444444;'>
        <h4>Model Overview</h4>
        <p>Our LSRM-GARCH (Long Short-Range Memory - Generalized Autoregressive Conditional Heteroskedasticity) 
        model combines the power of deep learning with traditional econometric techniques to provide 
        highly accurate volatility forecasts for the S&P 500 index.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Forecast controls
    col1, col2 = st.columns([2, 1])
    with col1:
        forecast_days = st.slider(
            "Forecast Horizon (Days):", 
            min_value=5, max_value=90, value=30, step=5
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh Forecast", type="primary"):
            st.rerun()
    
    # Get GARCH forecast data
    with st.spinner("Running LSRM-GARCH forecast..."):
        garch_data = make_api_request("/api/garch-predict", params={"forecast_horizon": forecast_days})
    
    if garch_data:
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            latest_vol = garch_data['predicted_volatility'][0] if garch_data['predicted_volatility'] else 0
            create_metric_card("Next Day Forecast", f"{latest_vol:.2f}", suffix="%")
        with col2:
            avg_vol = sum(garch_data['predicted_volatility']) / len(garch_data['predicted_volatility']) if garch_data['predicted_volatility'] else 0
            create_metric_card("Average Forecast", f"{avg_vol:.2f}", suffix="%")
        with col3:
            max_vol = max(garch_data['predicted_volatility']) if garch_data['predicted_volatility'] else 0
            create_metric_card("Peak Forecast", f"{max_vol:.2f}", suffix="%")
        
        # Main forecast chart
        plot_lsrm_garch_forecast(garch_data)
        
        # Forecast details
        st.markdown("### Forecast Details")
        df = pd.DataFrame({
            'Date': garch_data['dates'],
            'Predicted Volatility (%)': [f"{x:.4f}" for x in garch_data['predicted_volatility']]
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        
    else:
        st.error("Could not retrieve LSRM-GARCH forecast. Please check the backend connection.")


elif analysis_type == "S&P500 Volatility":
    st.title("📊 S&P500 Volatility Analysis (2000-Present)")
    st.markdown("---")
    
    sp500_data = make_api_request("/api/sp500")
    if sp500_data:
        plot_sp500_volatility(sp500_data['chart_data'])
        
        # Volatility statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            current_vol = sp500_data['latest_volatility']
            create_metric_card("Current Volatility", f"{current_vol:.2f}", suffix="%")
        with col2:
            avg_vol = sum(sp500_data['chart_data']['volatility']) / len(sp500_data['chart_data']['volatility'])
            create_metric_card("Average Volatility", f"{avg_vol:.2f}", suffix="%")
        with col3:
            max_vol = max(sp500_data['chart_data']['volatility'])
            create_metric_card("Peak Volatility", f"{max_vol:.2f}", suffix="%")
    else:
        st.error("Could not load S&P 500 volatility data.")

# elif analysis_type == "VIX Analysis":
#     st.title("😰 VIX Fear Index Analysis (2000-Present)")
#     st.markdown("---")
    
#     vix_data = make_api_request("/api/vix")
#     if vix_data:
#         plot_vix_overtime(vix_data['chart_data'])
        
#         # VIX statistics
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             current_vix = vix_data['latest_price']
#             create_metric_card("Current VIX", f"{current_vix:.2f}")
#         with col2:
#             avg_vix = sum(vix_data['chart_data']['prices']) / len(vix_data['chart_data']['prices'])
#             create_metric_card("Average VIX", f"{avg_vix:.2f}")
#         with col3:
#             max_vix = max(vix_data['chart_data']['prices'])
#             create_metric_card("Peak VIX", f"{max_vix:.2f}")
            
#         # VIX interpretation
#         st.markdown("### VIX Interpretation")
#         if current_vix < 15:
#             st.success("🟢 Low Volatility: Market is calm and stable")
#         elif current_vix < 25:
#             st.info("🟡 Moderate Volatility: Normal market conditions")
#         elif current_vix < 35:
#             st.warning("🟠 High Volatility: Increased market stress")
#         else:
#             st.error("🔴 Extreme Volatility: Market panic/fear")
#     else:
#         st.error("Could not load VIX data.")

# --- SMS Alert Section (always visible) ---
st.sidebar.markdown("---")
st.sidebar.title("🚨 Volatility Alerts")
st.sidebar.write("Get notified when LSRM-GARCH predicts high volatility")

with st.sidebar.form("alert_form"):
    phone_number = st.text_input("Phone Number", placeholder="+14155552671")
    threshold = st.number_input("Alert Threshold (%)", min_value=0.1, value=2.5, step=0.1)
    submit_button = st.form_submit_button(label='Set Alert')

    if submit_button:
        if not phone_number or not threshold:
            st.sidebar.error("Please enter valid phone number and threshold.")
        else:
            payload = {"phone_number": phone_number, "threshold": threshold}
            try:
                response = requests.post(f"{API_BASE_URL}/api/garch-alert", params=payload)
                response.raise_for_status()
                st.sidebar.success(f"✅ Alert set for {threshold}% threshold")
            except requests.exceptions.RequestException as e:
                error_detail = str(e)
                if e.response is not None:
                    error_detail = e.response.json().get('detail', error_detail)
                st.sidebar.error(f"Failed to set alert: {error_detail}")

# --- Footer ---
st.markdown("---")
st.markdown("<div style='text-align: center; color: #777;'>LSRM-GARCH Volatility Forecast Dashboard | Professional Financial Analytics</div>", unsafe_allow_html=True) 