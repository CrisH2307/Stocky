import pandas_datareader.data as web
import pandas as pd
import numpy as np
from arch import arch_model
import logging
from datetime import datetime
import pandas_datareader.data as web

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
        historical_vol = full_garch_fit.conditional_volatility * np.sqrt(252)  # Annualize

        # 2. Split data for training and testing
        split_index = int(len(returns) * 0.8)
        train_returns = returns.iloc[:split_index]
        
        # 3. Fit model on the training data ONLY
        train_garch = arch_model(train_returns, p=3, q=0)
        train_garch_fit = train_garch.fit(disp='off')
        
        # 4. Get in-sample (training) and out-of-sample (testing) predictions
        # The conditional_volatility from the fitted model gives us the in-sample predictions.
        train_pred_vol = train_garch_fit.conditional_volatility * np.sqrt(252)  # Annualize
        
        # To get the test prediction, we do a rolling forecast to follow historical patterns
        test_pred_vol = []
        test_dates = returns.index[split_index:]
        
        for i in range(len(test_dates)):
            # Use data up to split_index + i for each forecast
            current_train_data = returns.iloc[:split_index + i]
            if len(current_train_data) > 50:  # Ensure enough data for GARCH
                try:
                    current_garch = arch_model(current_train_data, p=3, q=0)
                    current_fit = current_garch.fit(disp='off')
                    # Forecast one step ahead
                    current_forecast = current_fit.forecast(horizon=1, reindex=False)
                    vol_pred = np.sqrt(current_forecast.variance.iloc[-1, 0]) * np.sqrt(252)  # Annualize
                    test_pred_vol.append(vol_pred)
                except:
                    # If forecast fails, use the last known volatility
                    if test_pred_vol:
                        test_pred_vol.append(test_pred_vol[-1])
                    else:
                        test_pred_vol.append(historical_vol.iloc[split_index + i - 1] if split_index + i > 0 else 0)
            else:
                # Not enough data, use historical volatility
                test_pred_vol.append(historical_vol.iloc[split_index + i] if split_index + i < len(historical_vol) else 0)
        
        test_pred_vol = pd.Series(test_pred_vol, index=test_dates)

        # 5. Forecast the next 7 days from the full model
        full_forecast = full_garch_fit.forecast(horizon=7, reindex=False)
        forecast_vol = np.sqrt(full_forecast.variance.iloc[-1].values) * np.sqrt(252)  # Annualize

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

    # Calculate error metrics using the same approach as the notebook
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
                train_mae = np.mean(np.abs(y_train_inv - train_pred_inv)) * 100
                train_rmse = np.sqrt(np.mean((y_train_inv - train_pred_inv)**2)) * 100
        
        # Calculate test metrics - compare predicted vs actual (like y_test_inv vs test_pred_inv)
        if test_vol is not None and len(test_vol) > 0:
            test_actual = historical_vol.loc[test_vol.index]
            # Remove any NaN values
            valid_mask = ~(np.isnan(test_vol) | np.isnan(test_actual))
            if np.sum(valid_mask) > 0:
                # Convert to numpy arrays for calculation
                y_test_inv = test_actual[valid_mask].values
                test_pred_inv = test_vol[valid_mask].values
                test_mae = np.mean(np.abs(y_test_inv - test_pred_inv)) * 100
                test_rmse = np.sqrt(np.mean((y_test_inv - test_pred_inv)**2)) * 100
            
        logger.info(f"Live error metrics for {ticker_symbol} - Train MAE: {train_mae:.4f}%, Train RMSE: {train_rmse:.4f}%, Test MAE: {test_mae:.4f}%, Test RMSE: {test_rmse:.4f}%")
    except Exception as e:
        logger.warning(f"Error calculating live metrics for {ticker_symbol}: {e}")

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
        "forecast_values": [v if np.isfinite(v) else None for v in forecast_values],
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "test_mae": test_mae,
        "test_rmse": test_rmse
    } 