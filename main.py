from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging

# --- Import Custom Modules ---
from sms_alert import send_sms
from garch_model import load_and_prepare_data, run_garch_forecast
from local_data_processor import load_excel_data, analyze_sp500_data, analyze_vix_data

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

# Storage for alerts. Key: phone_number, Value: threshold
garch_alert_rule = {} 

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
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 