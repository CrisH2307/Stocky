import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="Stock Volatility Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to match the screenshot
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background-color: #0E1117;
    }
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
        padding-top: 1rem;
    }
    /* Section header */
    .section-header {
        font-size: 1.75rem;
        font-weight: bold;
        color: #FFFFFF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    /* Custom metric card */
    .metric-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-card h4 {
        color: #8B949E;
        font-size: 1rem;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
    }
    .metric-card p {
        color: #C9D1D9;
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-card .delta {
        font-size: 1rem;
        font-weight: normal;
    }
    .delta-positive {
        color: #3FB950;
    }
    .delta-negative {
        color: #F85149;
    }
    .volatility-value {
        color: #FFA500; /* Orange for volatility */
    }
</style>
""", unsafe_allow_html=True)

# API base URL
API_BASE_URL = "http://localhost:8000"

# --- API Functions ---
def get_api_data(endpoint):
    """Generic function to fetch data from API"""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching {endpoint}: {response.text}")
            return None
    except requests.exceptions.RequestException:
        st.error(f"Could not connect to API. Is the backend running at {API_BASE_URL}?")
        return None

# --- Charting Functions ---
def create_price_chart(chart_data):
    """Create price and volume chart with dark theme"""
    if not chart_data or not chart_data.get('dates'):
        return None
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Scatter(x=chart_data['dates'], y=chart_data['prices'], mode='lines', name='Price', line=dict(color='#58A6FF', width=2)), row=1, col=1)
    fig.add_trace(go.Bar(x=chart_data['dates'], y=chart_data['volumes'], name='Volume', marker_color='#3FB950', opacity=0.6), row=2, col=1)
    
    fig.update_layout(
        height=500, showlegend=False, xaxis_rangeslider_visible=False,
        template='plotly_dark', margin=dict(t=20, b=0, l=0, r=0),
        yaxis=dict(title='Price (USD)', gridcolor='#30363D'),
        yaxis2=dict(title='Volume', gridcolor='#30363D'),
        xaxis=dict(gridcolor='#30363D')
    )
    return fig

def create_volatility_chart(chart_data):
    """Create volatility chart with dark theme"""
    if not chart_data or not chart_data.get('dates'):
        return None
        
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_data['dates'], y=chart_data['volatility'], mode='lines', name='Volatility', line=dict(color='#FFA500', width=2), fill='tozeroy', fillcolor='rgba(255, 165, 0, 0.1)'))
    
    fig.update_layout(
        height=400, template='plotly_dark', margin=dict(t=20, b=0, l=0, r=0),
        title_text="30-Day Rolling Volatility", title_x=0.5,
        yaxis=dict(title='Volatility', gridcolor='#30363D'),
        xaxis=dict(gridcolor='#30363D')
    )
    return fig

# --- UI Display Functions ---
def display_metric(label, value, delta=None):
    """Helper to display a single metric card using HTML/CSS"""
    delta_html = ""
    if delta is not None:
        delta_val, delta_pct = delta
        delta_class = "delta-positive" if delta_val >= 0 else "delta-negative"
        arrow = "↑" if delta_val >= 0 else "↓"
        delta_html = f'<p class="delta {delta_class}">{arrow} {delta_pct:.2f}%</p>'
    
    value_class = "volatility-value" if "volatility" in label.lower() else ""

    st.markdown(f"""
    <div class="metric-card">
        <h4>{label}</h4>
        <p class="{value_class}">{value}</p>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def display_stock_overview(symbol):
    """Display stock overview in the new UI style"""
    st.markdown(f'<h2 class="section-header">📊 {symbol} Stock Overview</h2>', unsafe_allow_html=True)
    
    stock_data = get_api_data(f"/api/stocks/{symbol}")
    chart_data = get_api_data(f"/api/stocks/{symbol}/chart")
    
    if stock_data:
        # Key metrics using new layout
        cols = st.columns(4)
        with cols[0]:
            display_metric("Current Price", f"${stock_data['current_price']}", (stock_data['price_change'], stock_data['price_change_pct']))
        with cols[1]:
            display_metric("Current Volatility", f"{stock_data['current_volatility']:.4f}")
        with cols[2]:
            display_metric("Volume", f"{stock_data['volume']:,}")
        with cols[3]:
            display_metric("52W High", f"${stock_data['high_52w']}")
            
        # Charts
        st.markdown('<h3 class="section-header" style="text-align: center; margin-top: 3rem;">Price and Volatility Charts</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns([3, 2])
        with col1:
            price_chart = create_price_chart(chart_data)
            if price_chart:
                st.plotly_chart(price_chart, use_container_width=True)
        with col2:
            volatility_chart = create_volatility_chart(chart_data)
            if volatility_chart:
                st.plotly_chart(volatility_chart, use_container_width=True)

def display_volatility_analysis(symbol):
    """Display detailed volatility analysis"""
    st.markdown(f'<h2 class="section-header">📈 {symbol} Volatility Analysis</h2>', unsafe_allow_html=True)
    volatility_data = get_api_data(f"/api/volatility/{symbol}")
    
    if volatility_data:
        cols = st.columns(4)
        with cols[0]:
            display_metric("Current Volatility", f"{volatility_data['current_volatility']:.4f}")
        with cols[1]:
            display_metric("Volatility Percentile", f"{volatility_data['volatility_percentile']:.1f}%")
        with cols[2]:
            display_metric("Vol of Vol", f"{volatility_data['vol_of_vol']:.4f}")
        with cols[3]:
            trend_icon = "📈" if volatility_data['volatility_trend'] == "increasing" else "📉"
            display_metric("Trend", f"{trend_icon} {volatility_data['volatility_trend'].title()}")
            
        chart_data = get_api_data(f"/api/stocks/{symbol}/chart")
        volatility_chart = create_volatility_chart(chart_data)
        if volatility_chart:
            st.plotly_chart(volatility_chart, use_container_width=True)

def display_market_overview():
    """Display market overview"""
    st.markdown('<h2 class="section-header">🌍 Market Overview</h2>', unsafe_allow_html=True)
    market_data = get_api_data("/api/market/overview")
    
    if market_data:
        indices = {"^GSPC": "S&P 500", "^DJI": "Dow Jones", "^IXIC": "NASDAQ", "^VIX": "VIX"}
        cols = st.columns(len(indices))
        
        for i, (symbol, name) in enumerate(indices.items()):
            with cols[i]:
                if symbol in market_data and "error" not in market_data[symbol]:
                    data = market_data[symbol]
                    st.metric(label=name, value=f"{data['current_price']:.2f}", delta=f"{data['price_change_pct']:.2f}%")
                else:
                    st.metric(label=name, value="N/A")

# Main app
def main():
    st.markdown('<h1 class="main-header">📈 Stock Volatility Dashboard</h1>', unsafe_allow_html=True)
    
    # --- Sidebar ---
    st.sidebar.title("🔍 Controls")
    default_symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    symbol_list = default_symbols + [st.sidebar.text_input("Or enter custom symbol:", "").upper()]
    symbol = st.sidebar.selectbox("Select Stock Symbol:", [s for s in symbol_list if s], index=0)
    
    analysis_type = st.sidebar.selectbox("Analysis Type:", ["Stock Overview", "Volatility Analysis", "Market Overview"])
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # --- Main content area ---
    if analysis_type == "Stock Overview":
        display_stock_overview(symbol)
    elif analysis_type == "Volatility Analysis":
        display_volatility_analysis(symbol)
    elif analysis_type == "Market Overview":
        display_market_overview()

if __name__ == "__main__":
    main() 