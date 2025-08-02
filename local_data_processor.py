import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_excel_data(file_path: str, data_name: str) -> pd.DataFrame:
    """
    A generic function to load and preprocess data from the provided Excel files.
    """
    try:
        df = pd.read_excel(file_path, skiprows=5)
        df.columns = ['Date', 'Close']
        df.dropna(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        df.sort_values('Date', inplace=True)
        df.set_index('Date', inplace=True)
        logger.info(f"Successfully loaded and processed {data_name} data from {file_path}.")
        return df
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}. Please ensure it's in the 'data/' directory.")
        return None
    except Exception as e:
        logger.error(f"An error occurred while loading {file_path}: {e}")
        return None

def analyze_sp500_data(df: pd.DataFrame) -> dict:
    """
    Analyzes the S&P 500 DataFrame to extract key metrics and historical data.
    """
    if df is None:
        return {}
    
    # Calculate returns and volatility
    df['Returns'] = 100 * df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(window=22).std() * np.sqrt(252) # Annualized volatility
    
    # Extract latest metrics
    latest_data = df.iloc[-1]
    previous_data = df.iloc[-2]
    
    price_change = latest_data['Close'] - previous_data['Close']
    price_change_pct = (price_change / previous_data['Close']) * 100

    # Clean data for JSON response
    df_cleaned = df.fillna(0)

    return {
        "latest_price": round(latest_data['Close'], 2),
        "price_change_pct": round(price_change_pct, 2),
        "latest_volatility": round(latest_data['Volatility'], 2),
        "chart_data": {
            "dates": df_cleaned.index.strftime('%Y-%m-%d').tolist(),
            "prices": df_cleaned['Close'].tolist(),
            "volatility": df_cleaned['Volatility'].tolist()
        }
    }

def analyze_vix_data(df: pd.DataFrame) -> dict:
    """
    Analyzes the VIX DataFrame to extract key metrics and historical data.
    """
    if df is None:
        return {}
        
    latest_data = df.iloc[-1]
    
    return {
        "latest_price": round(latest_data['Close'], 2),
        "chart_data": {
            "dates": df.index.strftime('%Y-%m-%d').tolist(),
            "prices": df['Close'].tolist(),
        }
    }

# Example usage for testing this module
if __name__ == "__main__":
    print("Testing local_data_processor...")
    
    # Test S&P 500 data
    sp500_df = load_excel_data('data/data_SP500.xlsx', 'S&P 500')
    if sp500_df is not None:
        sp500_analysis = analyze_sp500_data(sp500_df)
        print("\nS&P 500 Analysis Results:")
        print(f"  - Latest Price: {sp500_analysis['latest_price']}")
        print(f"  - Latest Volatility: {sp500_analysis['latest_volatility']}")
        print(f"  - Data points for chart: {len(sp500_analysis['chart_data']['dates'])}")

    # Test VIX data
    vix_df = load_excel_data('data/data_VIX.xlsx', 'VIX')
    if vix_df is not None:
        vix_analysis = analyze_vix_data(vix_df)
        print("\nVIX Analysis Results:")
        print(f"  - Latest Price: {vix_analysis['latest_price']}")
        print(f"  - Data points for chart: {len(vix_analysis['chart_data']['dates'])}") 