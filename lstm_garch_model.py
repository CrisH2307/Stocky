import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import logging
import time
import os
import joblib
from arch import arch_model

from local_data_processor import load_excel_data, analyze_sp500_data
from garch_model import load_and_prepare_data, run_garch_forecast

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = 'cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'model_cache.joblib')

_model_cache = {}

def _load_or_train_model():
    """
    Loads the trained model from the file cache if it exists.
    If not, it runs the full training pipeline and saves the result to the cache.
    """
    global _model_cache
    if _model_cache:
        return

    if os.path.exists(CACHE_FILE):
        logger.info(f"Found model cache file. Loading from {CACHE_FILE}...")
        _model_cache = joblib.load(CACHE_FILE)
        logger.info("Model loaded from cache successfully.")
        return

    logger.info("--- No cache file found. Starting one-time model training pipeline. ---")
    start_time = time.time()
    
    # Step 1: Prepare features
    returns = load_and_prepare_data('data/data_SP500.xlsx')
    # Using daily volatility, not annualized
    volatility = returns.rolling(window=22).std()
    volatility = volatility.dropna()
    
    # Run GARCH, but get daily vol, not annualized
    garch_fit = arch_model(returns, p=1, q=1).fit(disp='off')
    garch_predictions_series = garch_fit.conditional_volatility.loc[volatility.index]
    
    # Step 2: Create Hybrid DataFrame
    hybrid_df = pd.concat([
        returns.rename("log_return"),
        volatility.shift(1).rename("lagged_vol"),
        garch_predictions_series.rename("garch_forecast"),
        volatility.rename("target_vol")
    ], axis=1).dropna()
    
    # Step 3: Scale and create dataset
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(hybrid_df)
    
    look_back = 20
    def create_dataset(data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back - 1):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back, 3])
        return np.array(X), np.array(y)
    X, y = create_dataset(scaled_data, look_back)
    
    # Step 4: Train/Test Split
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Step 5: Build and Train LSTM
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(look_back, X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    logger.info(f"Starting LSTM model.fit() on {len(X_train)} samples...")
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # Create and cache a scaler for the target variable
    scaler_y = MinMaxScaler()
    scaler_y.min_, scaler_y.scale_ = scaler.min_[3], scaler.scale_[3]

    _model_cache = {
        'model': model,
        'scaler': scaler,
        'scaler_y': scaler_y,
        'hybrid_df': hybrid_df,
        'look_back': look_back
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    joblib.dump(_model_cache, CACHE_FILE)
    
    end_time = time.time()
    logger.info(f"--- Model training complete. Saved to cache. Took {end_time - start_time:.2f} seconds. ---")

def get_lstm_garch_performance_plot():
    """
    Generates the data for the performance plot (train/test vs actual).
    Ensures the model is trained first.
    """
    try:
        # Call the function to populate the global _model_cache
        _load_or_train_model()
        
        # Unpack directly from the global cache variable
        model = _model_cache['model']
        scaler_y = _model_cache['scaler_y']
        hybrid_df = _model_cache['hybrid_df']
        look_back = _model_cache['look_back']
        
        # Recreate datasets to get full train/test sets for plotting
        scaler = _model_cache['scaler']
        scaled_data = scaler.transform(hybrid_df)
        X, y = create_dataset(scaled_data, look_back)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Generate predictions
        train_pred_scaled = model.predict(X_train)
        test_pred_scaled = model.predict(X_test)
        
        train_pred = scaler_y.inverse_transform(train_pred_scaled)
        test_pred = scaler_y.inverse_transform(test_pred_scaled)
        
        # Format for plotting
        full_index = hybrid_df.index[look_back + 1:]
        train_pred_series = pd.Series(train_pred.flatten(), index=full_index[:split_index])
        test_pred_series = pd.Series(test_pred.flatten(), index=full_index[split_index:])
        
        # Get actual volatility from the original dataframe
        volatility = hybrid_df['target_vol']
        actual_vol_train = volatility.loc[train_pred_series.index]
        actual_vol_test = volatility.loc[test_pred_series.index]
        
        return {
            "train_dates": train_pred_series.index.strftime('%Y-%m-%d').tolist(),
            "train_pred": train_pred_series.tolist(),
            "train_actual": actual_vol_train.tolist(),
            "test_dates": test_pred_series.index.strftime('%Y-%m-%d').tolist(),
            "test_pred": test_pred_series.tolist(),
            "test_actual": actual_vol_test.tolist()
        }
    except Exception as e:
        logger.error(f"Error generating performance plot data: {e}", exc_info=True)
        return None

def get_7_day_lstm_garch_forecast():
    """
    Generates a 7-day ahead volatility forecast using the trained LSTM-GARCH model.
    """
    try:
        # Call the function to populate the global _model_cache
        _load_or_train_model()

        # Unpack directly from the global cache variable
        model = _model_cache['model']
        scaler = _model_cache['scaler']
        scaler_y = _model_cache['scaler_y']
        hybrid_df = _model_cache['hybrid_df']
        look_back = _model_cache['look_back']
        
        # Get the last sequence from the historical data
        scaled_data = scaler.transform(hybrid_df)
        last_sequence = scaled_data[-look_back:]
        
        future_predictions_scaled = []
        current_sequence = last_sequence.reshape(1, look_back, last_sequence.shape[1])

        # Iteratively predict the next 7 days
        for _ in range(7):
            # Predict the next step
            next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
            future_predictions_scaled.append(next_pred_scaled)
            
            # Create the new step to be appended
            new_step = current_sequence[0, -1, :].copy() 
            #new_step[3] = next_pred_scaled # Update the target_vol feature with the prediction
            new_step[1] = next_pred_scaled
            
            # Append the new step and remove the oldest one
            new_sequence_2d = np.append(current_sequence[0, 1:, :], [new_step], axis=0)
            current_sequence = new_sequence_2d.reshape(1, look_back, last_sequence.shape[1])
            
        # Inverse transform the predictions
        forecast = scaler_y.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
        
        # Create future dates starting from today
        start_date = pd.to_datetime('today').normalize()
        forecast_dates = pd.to_datetime([start_date + pd.DateOffset(days=i) for i in range(1, 8)])
        
        return {
            "dates": forecast_dates.strftime('%Y-%m-%d').tolist(),
            "forecast": forecast.flatten().tolist()
        }
    except Exception as e:
        logger.error(f"Error generating 7-day forecast: {e}", exc_info=True)
        return None
        
def create_dataset(data, look_back=20):
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back, 3])
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # For testing the module directly
    print("Running LSTM-GARCH model pipeline as a standalone test...")
    results = get_lstm_garch_performance_plot()
    if results:
        print("Successfully generated predictions.")
        print(f"Train predictions: {len(results['train_pred'])}")
        print(f"Test predictions: {len(results['test_pred'])}")
    else:
        print("Failed to generate predictions.") 