import pandas as pd
import logging
from ticker_analyzer import get_ticker_data
import numpy as np
from arch import arch_model

logger = logging.getLogger(__name__)

def parse_portfolio_string(portfolio_string: str):
    """
    Parses a string like 'AAPL 0.6, MSFT 0.4' into a dictionary.
    Returns a dictionary of tickers and weights, and validates if weights sum to 1.0.
    """
    portfolio = {}
    try:
        parts = [p.strip() for p in portfolio_string.split(',')]
        if not any(parts):
            return None, "Portfolio string is empty."

        for part in parts:
            split_part = part.split()
            if len(split_part) != 2:
                return None, f"Invalid format for part: '{part}'. Use 'TICKER WEIGHT'."
            ticker, weight_str = split_part
            weight = float(weight_str)
            if not (0 < weight <= 1):
                return None, "Each weight must be between 0 and 1."
            portfolio[ticker.upper()] = weight
        
        total_weight = sum(portfolio.values())
        if not (0.99 <= total_weight <= 1.01):
            return None, f"Portfolio weights must sum to 1.0 (currently {total_weight:.2f})."

        return portfolio, None
    except (ValueError, IndexError) as e:
        logger.error(f"Error parsing portfolio string: {portfolio_string}. Error: {e}")
        return None, "Invalid format. Please use format like 'AAPL 0.6, MSFT 0.4'."

def analyze_portfolio(portfolio_string: str):
    """
    Analyzes the 7-day volatility forecast of a portfolio and its constituent stocks.
    """
    portfolio, error = parse_portfolio_string(portfolio_string)
    if error:
        return {"error": error}

    individual_forecasts = {}
    last_date = None

    for ticker in portfolio.keys():
        history = get_ticker_data(ticker)
        if history is None:
            return {"error": f"Could not fetch data for ticker: {ticker}"}
        
        returns = 100 * history['Close'].pct_change().dropna()
        if returns.empty:
            return {"error": f"Could not calculate returns for {ticker}."}
        
        try:
            garch = arch_model(returns, p=3, q=0)
            garch_fit = garch.fit(disp='off')
            forecast = garch_fit.forecast(horizon=7, reindex=False).variance.iloc[-1]
            individual_forecasts[ticker] = np.sqrt(forecast) * np.sqrt(252)  # Annualize
            if last_date is None or returns.index[-1] > last_date:
                last_date = returns.index[-1]
        except Exception as e:
            logger.error(f"GARCH failed for {ticker}: {e}")
            return {"error": f"Volatility forecast failed for {ticker}."}

    if not individual_forecasts or last_date is None:
        return {"error": "Could not generate any forecasts."}

    # Combine forecasts
    df_forecasts = pd.DataFrame(individual_forecasts)
    portfolio_weights = pd.Series(portfolio)
    # Ensure weights match the columns
    aligned_weights = portfolio_weights.reindex(df_forecasts.columns).fillna(0)
    portfolio_forecast = df_forecasts.dot(aligned_weights)

    # Create forecast dates, starting from today.
    start_date = pd.to_datetime('today').normalize()
    forecast_dates = pd.date_range(start=start_date, periods=7)

    # Prepare individual volatilities for JSON output
    individual_vol_data = {}
    for ticker, forecast_series in df_forecasts.items():
        individual_vol_data[ticker] = {
            "dates": forecast_dates.strftime('%Y-%m-%d').tolist(),
            "volatility": [v if np.isfinite(v) else None for v in forecast_series.tolist()]
        }
    
    return {
        "portfolio_forecast": {
            "dates": forecast_dates.strftime('%Y-%m-%d').tolist(),
            "volatility": [v if np.isfinite(v) else None for v in portfolio_forecast.tolist()]
        },
        "individual_forecasts": individual_vol_data
    }