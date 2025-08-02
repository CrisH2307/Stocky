import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import time
import os
import joblib
from arch import arch_model
import math

from local_data_processor import load_excel_data, analyze_sp500_data
from garch_model import load_and_prepare_data, run_garch_forecast

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

    # === Step 1: Prepare features ===
    returns = load_and_prepare_data('data/data_SP500.xlsx')
    volatility = returns.rolling(window=22).std()
    volatility = volatility.dropna()
    garch_fit = arch_model(returns, p=1, q=1).fit(disp='off')
    garch_std_aligned = garch_fit.conditional_volatility.loc[volatility.index]

    # === Step 2: Merge input data ===
    hybrid_df = pd.concat([
        returns.rename("log_return"),
        volatility.shift(1).rename("lagged_vol"),
        garch_std_aligned.rename("garch_forecast"),
        volatility.rename("target_vol")
    ], axis=1).dropna()

    # === Step 3: Normalize data ===
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(hybrid_df)

    # === Step 4: Create dataset ===
    def create_dataset(data, look_back=20):
        X, y = [], []
        for i in range(len(data) - look_back - 1):
            X.append(data[i:i + look_back])
            y.append(data[i + look_back, 1])  # column 1 = lagged_vol (target_vol)
        return np.array(X), np.array(y)

    look_back = 20
    X, y = create_dataset(scaled, look_back)

    # === Step 5: Split into train/test ===
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # === Step 6: Build the LSTM model with final hyperparameters ===
    model = Sequential()
    model.add(LSTM(128, activation='tanh', return_sequences=True, input_shape=(look_back, X.shape[2])))
    model.add(Dropout(0.1))
    model.add(LSTM(128, activation='tanh'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse')

    # === Step 7: Train the model ===
    model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=0)

    # === Step 8: Predict and inverse scale ===
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    scaler_y = MinMaxScaler()
    scaler_y.min_, scaler_y.scale_ = scaler.min_[1], scaler.scale_[1]
    train_pred_inv = scaler_y.inverse_transform(y_train_pred)
    test_pred_inv = scaler_y.inverse_transform(y_test_pred)
    y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # === Step 9: Rebuild full timeline ===
    full_index = hybrid_df.index[look_back + 1:]
    train_pred_series = pd.Series(train_pred_inv.flatten(), index=full_index[:split])
    test_pred_series = pd.Series(test_pred_inv.flatten(), index=full_index[split:])

    _model_cache = {
        'model': model,
        'scaler': scaler,
        'scaler_y': scaler_y,
        'hybrid_df': hybrid_df,
        'look_back': look_back,
        'train_pred_series': train_pred_series,
        'test_pred_series': test_pred_series,
        'y_train_inv': y_train_inv,
        'y_test_inv': y_test_inv,
        'split': split,
        'full_index': full_index
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
        _load_or_train_model()
        model = _model_cache['model']
        scaler_y = _model_cache['scaler_y']
        hybrid_df = _model_cache['hybrid_df']
        look_back = _model_cache['look_back']
        train_pred_series = _model_cache['train_pred_series']
        test_pred_series = _model_cache['test_pred_series']
        split = _model_cache['split']
        full_index = _model_cache['full_index']
        y_train_inv = _model_cache['y_train_inv']
        y_test_inv = _model_cache['y_test_inv']
        train_pred_inv = _model_cache['train_pred_series'].values.reshape(-1, 1)
        test_pred_inv = _model_cache['test_pred_series'].values.reshape(-1, 1)
        
        volatility = hybrid_df['target_vol']
        actual_vol_train = volatility.loc[train_pred_series.index]
        actual_vol_test = volatility.loc[test_pred_series.index]
        
        # Calculate error metrics in percentage format (multiply by 100)
        train_mae = mean_absolute_error(y_train_inv, train_pred_inv) * 100
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv)) * 100
        test_mae = mean_absolute_error(y_test_inv, test_pred_inv) * 100
        test_rmse = np.sqrt(mean_squared_error(y_test_inv, test_pred_inv)) * 100
        
        # Debug: Print the actual error values
        logger.info(f"Debug - Train MAE: {train_mae:.6f}%, Train RMSE: {train_rmse:.6f}%")
        logger.info(f"Debug - Test MAE: {test_mae:.6f}%, Test RMSE: {test_rmse:.6f}%")
        
        return {
            "train_dates": train_pred_series.index.strftime('%Y-%m-%d').tolist(),
            "train_pred": (train_pred_series * math.sqrt(252)).tolist(),
            "train_actual": (actual_vol_train * math.sqrt(252)).tolist(),
            "test_dates": test_pred_series.index.strftime('%Y-%m-%d').tolist(),
            "test_pred": (test_pred_series * math.sqrt(252)).tolist(),
            "test_actual": (actual_vol_test * math.sqrt(252)).tolist(),
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse
        }
    except Exception as e:
        logger.error(f"Error generating performance plot data: {e}", exc_info=True)
        return None

def get_7_day_lstm_garch_forecast():
    """
    Generates a 7-day ahead volatility forecast using the trained LSTM-GARCH model.
    """
    try:
        _load_or_train_model()
        model = _model_cache['model']
        scaler = _model_cache['scaler']
        scaler_y = _model_cache['scaler_y']
        hybrid_df = _model_cache['hybrid_df']
        look_back = _model_cache['look_back']
        scaled_data = scaler.transform(hybrid_df)
        last_sequence = scaled_data[-look_back:]
        future_predictions_scaled = []
        current_sequence = last_sequence.reshape(1, look_back, last_sequence.shape[1])
        for _ in range(7):
            next_pred_scaled = model.predict(current_sequence, verbose=0)[0, 0]
            future_predictions_scaled.append(next_pred_scaled)
            new_step = current_sequence[0, -1, :].copy()
            new_step[1] = next_pred_scaled  # update lagged_vol with prediction
            new_sequence_2d = np.append(current_sequence[0, 1:, :], [new_step], axis=0)
            current_sequence = new_sequence_2d.reshape(1, look_back, last_sequence.shape[1])
        forecast = scaler_y.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
        import pandas as pd
        start_date = pd.to_datetime('today').normalize()
        forecast_dates = pd.to_datetime([start_date + pd.DateOffset(days=i) for i in range(1, 8)])
        return {
            "dates": forecast_dates.strftime('%Y-%m-%d').tolist(),
            "forecast": (forecast.flatten() * math.sqrt(252)).tolist()
        }
    except Exception as e:
        logger.error(f"Error generating 7-day forecast: {e}", exc_info=True)
        return None 