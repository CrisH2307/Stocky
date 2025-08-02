import pandas as pd
from arch import arch_model
import numpy as np
import logging
from typing import List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_prepare_data(file_path: str) -> pd.Series:
    """
    Loads S&P 500 data from an Excel file, calculates percentage returns, and returns them.
    This version correctly handles the specific format of the data_SP500.xlsx file.
    """
    try:
        # This logic correctly reads the excel file by skipping headers,
        # not by assuming a sheet name.
        df = pd.read_excel(file_path, skiprows=5)
        df.columns = ['Date', 'Close']
        df.dropna(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)

        # Match the notebook's calculation: 100 * pct_change()
        returns = 100 * df['Close'].pct_change().dropna()
        logger.info(f"Data loaded successfully from {file_path}. Shape of returns: {returns.shape}")
        return returns
    except Exception as e:
        logger.error(f"An error occurred while loading data from {file_path}: {e}", exc_info=True)
        return None

def run_garch_forecast(returns: pd.Series, forecast_horizon: int = 30) -> Tuple[List[float], List[str]]:
    """
    Performs a rolling GARCH(2,2) forecast on a series of returns.

    Args:
        returns (pd.Series): The time series of asset returns.
        forecast_horizon (int): The number of steps to forecast ahead.

    Returns:
        Tuple[List[float], List[str]]: A tuple containing the list of predicted
        volatility values and the list of corresponding dates.
    """
    if returns is None or returns.empty:
        logger.warning("Cannot run GARCH forecast on empty returns data.")
        return [], []
        
    rolling_predictions = []
    test_size = forecast_horizon
    train_size = len(returns) - test_size

    if train_size < 1:
        logger.error("Not enough data to perform a rolling forecast.")
        return [], []

    logger.info(f"Starting GARCH rolling forecast for {test_size} steps...")
    
    for i in range(test_size):
        train_data = returns[:train_size + i]
        
        # Define the GARCH(2,2) model
        model = arch_model(train_data, p=2, q=2, vol='Garch', dist='Normal')
        
        # Fit the model, suppressing output for cleaner logs
        model_fit = model.fit(disp='off')
        
        # Forecast one step ahead
        pred = model_fit.forecast(horizon=1)
        
        # Get the standard deviation (volatility) from the forecast variance
        # The result is already scaled as per the returns (i.e., in percent)
        forecast_vol = np.sqrt(pred.variance.values[-1, :][0]) * np.sqrt(252)  # Annualize
        rolling_predictions.append(forecast_vol)

    logger.info("GARCH rolling forecast completed.")
    
    # Get the dates for the prediction period
    prediction_dates = returns.index[train_size:].strftime('%Y-%m-%d').tolist()
    
    return rolling_predictions, prediction_dates

# Example of how to run the full process
if __name__ == "__main__":
    data_filepath = 'data/data_SP500.xlsx'
    
    print("1. Loading and preparing data...")
    sp500_returns = load_and_prepare_data(data_filepath)
    
    if sp500_returns is not None:
        print("\n2. Running GARCH forecast...")
        predictions, dates = run_garch_forecast(sp500_returns, forecast_horizon=50)
        
        if predictions:
            print(f"\n✅ Forecast successful. Predicted {len(predictions)} steps.")
            # Print the last 5 predictions as a sample
            for date, pred_vol in list(zip(dates, predictions))[-5:]:
                print(f"   - Date: {date}, Predicted Volatility: {pred_vol:.4f}%")
        else:
            print("❌ Forecast failed. Check logs for details.")
    else:
        print("❌ Data loading failed. Cannot proceed with forecast.") 