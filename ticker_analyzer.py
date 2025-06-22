import pandas_datareader.data as web
import pandas as pd
import numpy as np
from arch import arch_model
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def get_ticker_data(ticker_symbol: str):
    """Fetches historical data for a given ticker from Stooq."""
    try:
        start = datetime(2015, 1, 1)
        end = datetime.today()
        history = web.DataReader(ticker_symbol.upper(), 'stooq', start=start, end=end)
        if history.empty:
            logger.warning(f"No data found for ticker: {ticker_symbol}")
            return None
        # Data from stooq is reverse-chronological, so we need to sort it.
        history.sort_index(inplace=True)
        logger.info(f"Successfully fetched data for {ticker_symbol} from stooq")
        return history
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol} from stooq: {e}")
        return None

def run_garch_analysis_for_ticker(returns: pd.Series):
    """
    Fits a GARCH(3,0) model on the returns of a specific stock,
    providing historical, in-sample, and out-of-sample volatility.
    This is a simplified version for consistency.
    """
    try:
        # 1. Get historical volatility for the full series (for comparison)
        full_garch = arch_model(returns, p=3, q=0)
        full_garch_fit = full_garch.fit(disp='off')
        historical_vol = full_garch_fit.conditional_volatility

        # 2. Split data for training and testing
        split_index = int(len(returns) * 0.8)
        train_returns = returns.iloc[:split_index]
        
        # 3. Fit model on the training data ONLY
        train_garch = arch_model(train_returns, p=3, q=0)
        train_garch_fit = train_garch.fit(disp='off')
        
        # 4. Get in-sample (training) and out-of-sample (testing) predictions
        # The conditional_volatility from the fitted model gives us the in-sample predictions.
        train_pred_vol = train_garch_fit.conditional_volatility
        
        # To get the test prediction, we forecast from the end of the training period.
        # This is a static, one-time forecast, which is simpler and faster.
        forecast = train_garch_fit.forecast(
            horizon=len(returns) - split_index, 
            reindex=False
        )
        test_pred_vol = np.sqrt(forecast.variance.iloc[-1])
        test_pred_vol.index = returns.index[split_index:]

        # 5. Forecast the next 7 days from the full model
        full_forecast = full_garch_fit.forecast(horizon=7, reindex=False)
        forecast_vol = np.sqrt(full_forecast.variance.iloc[-1].values)

        return historical_vol, train_pred_vol, test_pred_vol, forecast_vol.tolist()

    except Exception as e:
        logger.error(f"Error running GARCH(3,0) analysis for custom ticker: {e}", exc_info=True)
        return None, None, None, None

def analyze_ticker(ticker_symbol: str):
    """
    Orchestrator function to fetch, process, and analyze a given stock ticker.
    """
    history = get_ticker_data(ticker_symbol)
    if history is None:
        return None

    # Calculate percentage returns
    returns = 100 * history['Close'].pct_change().dropna()

    # Get volatility analysis
    historical_vol, train_vol, test_vol, forecast_values = run_garch_analysis_for_ticker(returns)

    if historical_vol is None:
        return {"error": "Could not generate volatility analysis."}

    # Prepare dates
    forecast_dates = pd.date_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=7)

    return {
        "ticker": ticker_symbol,
        "returns_dates": returns.index.strftime('%Y-%m-%d').tolist(),
        "returns_values": [v if np.isfinite(v) else None for v in returns.tolist()],
        "historical_vol_dates": historical_vol.index.strftime('%Y-%m-%d').tolist(),
        "historical_vol_values": [v if np.isfinite(v) else None for v in historical_vol.tolist()],
        "train_vol_dates": train_vol.index.strftime('%Y-%m-%d').tolist(),
        "train_vol_values": [v if np.isfinite(v) else None for v in train_vol.tolist()],
        "test_vol_dates": test_vol.index.strftime('%Y-%m-%d').tolist(),
        "test_vol_values": [v if np.isfinite(v) else None for v in test_vol.tolist()],
        "forecast_dates": forecast_dates.strftime('%Y-%m-%d').tolist(),
        "forecast_values": [v if np.isfinite(v) else None for v in forecast_values]
    } 