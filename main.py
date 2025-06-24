import sys
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import os # Import the os module
import subprocess
import time
import webbrowser

# --- Import Custom Modules ---
from sms_alert import send_sms
from garch_model import load_and_prepare_data, run_garch_forecast
from local_data_processor import load_excel_data, analyze_sp500_data, analyze_vix_data
from lstm_garch_model import get_lstm_garch_performance_plot, get_7_day_lstm_garch_forecast
from ticker_analyzer import analyze_ticker
from portfolio_analyzer import analyze_portfolio
from pydantic import BaseModel

# --- Force delete cache to ensure model retrains with new logic ---
# CACHE_FILE = 'cache/model_cache.joblib'
# if os.path.exists(CACHE_FILE):
#     try:
#         os.remove(CACHE_FILE)
#         print("--- DELETED OLD MODEL CACHE to apply new logic. Model will retrain now. ---")
#     except Exception as e:
#         print(f"--- Error deleting cache file, please delete manually: {e} ---")
# --- End of one-time cache deletion ---

# --- App Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="S&P 500 Volatility Forecast API",
    description="An API to serve S&P 500 data analysis and GARCH volatility forecasts.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Caching and Storage ---
# Simple caching for a hackathon to avoid re-reading files and re-running analysis.
sp500_data_cache = None
vix_data_cache = None
garch_results_cache = {}
lstm_garch_cache = None # Cache for the new performance data
lstm_7_day_forecast_cache = None # Cache for the 7-day forecast
portfolio_cache = {} # Cache for portfolio analysis results

# Storage for alerts. Key: phone_number, Value: threshold
garch_alert_rule = {} 

# --- Pydantic Models ---
class PortfolioPayload(BaseModel):
    portfolio_string: str

# --- Background Task for Alerts ---
def check_garch_and_alert(phone_number: str, threshold: float):
    """
    Runs a GARCH forecast and sends an SMS if the next day's predicted
    volatility exceeds the user-defined threshold.
    """
    logger.info(f"BACKGROUND: Checking GARCH forecast against threshold {threshold} for {phone_number}.")
    try:
        # We always want to run a fresh forecast for alerts. We forecast 1 step ahead.
        returns = load_and_prepare_data('data/data_SP500.xlsx')
        if returns is None:
            logger.error("BACKGROUND: Failed to load data for GARCH alert check.")
            return

        predictions, _ = run_garch_forecast(returns, forecast_horizon=1)
        
        if predictions:
            predicted_vol = predictions[0]
            if predicted_vol > threshold:
                logger.info(f"ALERT: GARCH predicted volatility ({predicted_vol:.4f}) for tomorrow exceeds threshold ({threshold}). Sending SMS.")
                message = f"📈 S&P 500 VOLATILITY ALERT: Tomorrow's predicted volatility is {predicted_vol:.4f}%, which is above your threshold of {threshold:.2f}%."
                send_sms(to_number=phone_number, message_body=message)
            else:
                logger.info(f"OK: GARCH predicted volatility ({predicted_vol:.4f}) is within threshold ({threshold}). No alert sent.")
    except Exception as e:
        logger.error(f"BACKGROUND: Error during GARCH alert check: {e}")

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to the S&P 500 Volatility Forecast API"}


@app.get("/api/sp500")
async def get_sp500_analysis():
    """Returns a full analysis of the S&P 500 data from the local Excel file."""
    global sp500_data_cache
    if sp500_data_cache is None:
        logger.info("No S&P 500 cache found. Loading and analyzing data...")
        df = load_excel_data('data/data_SP500.xlsx', 'S&P 500')
        if df is None:
            raise HTTPException(status_code=500, detail="Could not load S&P 500 data file.")
        sp500_data_cache = analyze_sp500_data(df)
    return sp500_data_cache


@app.get("/api/vix")
async def get_vix_analysis():
    """Returns a full analysis of the VIX data from the local Excel file."""
    global vix_data_cache
    if vix_data_cache is None:
        logger.info("No VIX cache found. Loading and analyzing data...")
        df = load_excel_data('data/data_VIX.xlsx', 'VIX')
        if df is None:
            raise HTTPException(status_code=500, detail="Could not load VIX data file.")
        vix_data_cache = analyze_vix_data(df)
    return vix_data_cache


@app.get("/api/ticker/{ticker_symbol}")
async def get_ticker_analysis(ticker_symbol: str):
    """
    Analyzes a given stock ticker using live data from stooq, returning its
    daily returns and a GARCH(3,0) volatility forecast.
    """
    logger.info(f"Received request to analyze ticker: {ticker_symbol}")
    analysis_result = analyze_ticker(ticker_symbol)
    if analysis_result is None:
        raise HTTPException(status_code=404, detail=f"Could not retrieve data for ticker '{ticker_symbol}'. Is it a valid symbol on stooq?")
    if "error" in analysis_result:
        raise HTTPException(status_code=500, detail=analysis_result["error"])
    return analysis_result


@app.get("/api/lstm-garch-performance")
async def get_lstm_garch_performance_data():
    """
    Returns the train/test predictions and actuals from the LSTM-GARCH model.
    The result is cached to avoid re-training the model on every request.
    """
    # global lstm_garch_cache
    # if lstm_garch_cache is None:
    #     logger.info("No LSTM-GARCH cache found. Running model pipeline...")
        # lstm_garch_cache = get_lstm_garch_performance_plot()
        # if lstm_garch_cache is None:
        #     raise HTTPException(status_code=500, detail="Failed to run LSTM-GARCH model pipeline.")
    lstm_garch_cache = get_lstm_garch_performance_plot()
    if lstm_garch_cache is None:
        raise HTTPException(status_code=500, detail="Failed to run LSTM-GARCH model pipeline.")
    return lstm_garch_cache
    

@app.get("/api/lstm-garch-7-day-forecast")
async def get_7_day_forecast():
    """
    Returns a 7-day ahead volatility forecast from the LSTM-GARCH model.
    The result is cached to avoid re-running the model pipeline.
    """
    # global lstm_7_day_forecast_cache
    # if lstm_7_day_forecast_cache is None:
    #     logger.info("No 7-day forecast cache found. Running forecast pipeline...")
    #     lstm_7_day_forecast_cache = get_7_day_lstm_garch_forecast()
    #     if lstm_7_day_forecast_cache is None:
    #         raise HTTPException(status_code=500, detail="Failed to generate 7-day forecast.")
    # return lstm_7_day_forecast_cache

    lstm_7_day_forecast_cache = get_7_day_lstm_garch_forecast()
    if lstm_7_day_forecast_cache is None:
        raise HTTPException(status_code=500, detail="Failed to generate 7-day forecast.")
    return lstm_7_day_forecast_cache

@app.get("/api/garch-predict")
async def get_garch_prediction(forecast_horizon: int = 30, background_tasks: BackgroundTasks = None):
    """
    Runs the GARCH(2,2) rolling forecast. Results are cached.
    Also triggers a background alert check if a rule is set.
    """
    global garch_results_cache
    if forecast_horizon in garch_results_cache:
        logger.info(f"Returning cached GARCH results for horizon {forecast_horizon}.")
        return garch_results_cache[forecast_horizon]

    logger.info(f"Running new GARCH forecast for horizon {forecast_horizon}...")
    returns = load_and_prepare_data('data/data_SP500.xlsx')
    if returns is None:
        raise HTTPException(status_code=500, detail="Failed to load data for GARCH model.")

    predictions, dates = run_garch_forecast(returns, forecast_horizon=forecast_horizon)
    if not predictions:
        raise HTTPException(status_code=500, detail="GARCH model failed to generate a forecast.")

    result = {"dates": dates, "predicted_volatility": predictions}
    garch_results_cache[forecast_horizon] = result
    
    # If an alert is set, run a check in the background
    if garch_alert_rule and background_tasks:
        phone = garch_alert_rule.get("phone_number")
        thresh = garch_alert_rule.get("threshold")
        if phone and thresh:
            background_tasks.add_task(check_garch_and_alert, phone, thresh)
            
    return result

@app.post("/analyze_portfolio/")
async def analyze_portfolio_endpoint(payload: PortfolioPayload):
    """
    Analyzes a portfolio of stocks given a string of tickers and weights.
    Caches the result to avoid re-fetching and re-calculating.
    """
    global portfolio_cache
    portfolio_string = payload.portfolio_string
    
    # Check cache first
    if portfolio_string in portfolio_cache:
        logger.info(f"Returning cached portfolio analysis for: {portfolio_string}")
        return portfolio_cache[portfolio_string]
        
    logger.info(f"Running new portfolio analysis for: {portfolio_string}")
    analysis_result = analyze_portfolio(portfolio_string)
    
    if "error" in analysis_result:
        raise HTTPException(status_code=400, detail=analysis_result["error"])
        
    portfolio_cache[portfolio_string] = analysis_result
    return analysis_result

@app.post("/api/garch-alert")
async def set_garch_alert(threshold: float, phone_number: str):
    """
    Sets or updates the volatility alert rule for the GARCH forecast.
    Note: This simplified system supports only one alert rule at a time.
    """
    if not phone_number or not threshold > 0:
        raise HTTPException(status_code=400, detail="A valid phone number and positive threshold are required.")
    
    garch_alert_rule["phone_number"] = phone_number
    garch_alert_rule["threshold"] = threshold
    
    logger.info(f"GARCH alert rule updated: Threshold={threshold}, Phone={phone_number}")
    return {"message": "GARCH volatility alert has been set successfully."}

if __name__ == "__main__":
        # Start FastAPI backend
    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--reload"])

    # Wait a bit to ensure backend is up
    time.sleep(2)

    # Start Streamlit frontend
    frontend = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard.py"])

    # Optionally, open the Streamlit app in the browser
    # time.sleep(2)
    # webbrowser.open("http://localhost:8501")

    # Wait for both processes to finish
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        backend.terminate()
        frontend.terminate()
