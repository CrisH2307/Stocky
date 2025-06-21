from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Volatility Dashboard API",
    description="Real-time stock data and volatility analysis for hackathon",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StockAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def get_stock_data(self, symbol: str, period: str = "1y") -> dict:
        """Fetch stock data and calculate basic metrics"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                logger.error(f"No data found for {symbol}")
                raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
            
            # Ensure we have enough data for calculations
            if len(hist) < 2:
                logger.error(f"Insufficient data for {symbol}")
                raise HTTPException(status_code=400, detail=f"Insufficient data for {symbol}")
            
            # Calculate returns
            hist['Returns'] = hist['Close'].pct_change()
            
            # Calculate volatility (rolling 30-day)
            hist['Volatility'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252)
            
            # Basic metrics
            current_price = hist['Close'].iloc[-1]
            price_change = hist['Close'].iloc[-1] - hist['Close'].iloc[-2]
            price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
            
            # Volatility metrics - handle NaN values
            current_volatility = hist['Volatility'].iloc[-1]
            if pd.isna(current_volatility):
                current_volatility = hist['Volatility'].mean()
            
            avg_volatility = hist['Volatility'].mean()
            if pd.isna(avg_volatility):
                avg_volatility = 0.0
            
            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "price_change": round(price_change, 2),
                "price_change_pct": round(price_change_pct, 2),
                "current_volatility": round(current_volatility, 4),
                "avg_volatility": round(avg_volatility, 4),
                "volume": int(hist['Volume'].iloc[-1]),
                "high_52w": round(hist['High'].max(), 2),
                "low_52w": round(hist['Low'].min(), 2),
                "data_points": len(hist)
            }
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            # Return a more specific error message
            raise HTTPException(status_code=500, detail=f"Error processing data for {symbol}: {str(e)}")

    def calculate_volatility_metrics(self, symbol: str, window: int = 30) -> dict:
        """Calculate detailed volatility metrics"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1y")
            
            if hist.empty:
                logger.error(f"No data found for {symbol}")
                raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
            
            if len(hist) < window:
                logger.error(f"Insufficient data for {symbol} (need {window} days, got {len(hist)})")
                raise HTTPException(status_code=400, detail=f"Insufficient data for volatility calculation")
            
            # Calculate returns
            returns = hist['Close'].pct_change().dropna()
            
            if len(returns) < window:
                logger.error(f"Insufficient returns data for {symbol}")
                raise HTTPException(status_code=400, detail=f"Insufficient returns data for volatility calculation")
            
            # Volatility calculations
            rolling_vol = returns.rolling(window=window).std() * np.sqrt(252)
            
            # Remove NaN values
            rolling_vol = rolling_vol.dropna()
            
            if len(rolling_vol) == 0:
                logger.error(f"No valid volatility data for {symbol}")
                raise HTTPException(status_code=400, detail=f"No valid volatility data")
            
            # Volatility of volatility
            vol_of_vol = rolling_vol.rolling(window=window).std()
            vol_of_vol = vol_of_vol.dropna()
            
            # Percentile ranks
            current_vol = rolling_vol.iloc[-1]
            vol_percentile = (rolling_vol < current_vol).mean() * 100
            
            # Handle NaN values
            if pd.isna(current_vol):
                current_vol = rolling_vol.mean()
            if pd.isna(vol_percentile):
                vol_percentile = 50.0
            if len(vol_of_vol) > 0 and not pd.isna(vol_of_vol.iloc[-1]):
                current_vol_of_vol = vol_of_vol.iloc[-1]
            else:
                current_vol_of_vol = 0.0
            
            return {
                "symbol": symbol,
                "current_volatility": round(current_vol, 4),
                "volatility_percentile": round(vol_percentile, 1),
                "vol_of_vol": round(current_vol_of_vol, 4),
                "min_volatility": round(rolling_vol.min(), 4),
                "max_volatility": round(rolling_vol.max(), 4),
                "avg_volatility": round(rolling_vol.mean(), 4),
                "volatility_trend": "increasing" if len(rolling_vol) >= 5 and rolling_vol.iloc[-1] > rolling_vol.iloc[-5] else "decreasing"
            }
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error calculating volatility for {symbol}: {str(e)}")

    def get_portfolio_risk(self, symbols: List[str]) -> dict:
        """Calculate portfolio risk metrics"""
        try:
            # Get data for all symbols
            portfolio_data = {}
            returns_data = {}
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="1y")
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    returns_data[symbol] = returns
                    portfolio_data[symbol] = {
                        "current_price": hist['Close'].iloc[-1],
                        "returns": returns.mean() * 252,  # Annualized
                        "volatility": returns.std() * np.sqrt(252)
                    }
            
            if not returns_data:
                raise HTTPException(status_code=400, detail="No valid symbols provided")
            
            # Create returns matrix
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Portfolio metrics (assuming equal weights)
            weights = np.array([1/len(symbols)] * len(symbols))
            portfolio_return = sum(portfolio_data[sym]["returns"] * weights[i] for i, sym in enumerate(symbols))
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(correlation_matrix * np.outer([portfolio_data[sym]["volatility"] for sym in symbols], [portfolio_data[sym]["volatility"] for sym in symbols]), weights)))
            
            # Sharpe ratio (assuming risk-free rate of 2%)
            risk_free_rate = 0.02
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0
            
            return {
                "portfolio_return": round(portfolio_return * 100, 2),
                "portfolio_volatility": round(portfolio_vol * 100, 2),
                "sharpe_ratio": round(sharpe_ratio, 3),
                "correlation_matrix": correlation_matrix.round(3).to_dict(),
                "holdings": portfolio_data
            }
        except Exception as e:
            logger.error(f"Error calculating portfolio risk: {str(e)}")
            raise HTTPException(status_code=500, detail="Error calculating portfolio risk")

# Initialize analyzer
analyzer = StockAnalyzer()

@app.get("/")
async def root():
    return {"message": "Stock Volatility Dashboard API", "status": "running"}

@app.get("/api/stocks/{symbol}")
async def get_stock_data(symbol: str, period: str = "1y"):
    """Get stock data and basic metrics"""
    return analyzer.get_stock_data(symbol.upper(), period)

@app.get("/api/volatility/{symbol}")
async def get_volatility_metrics(symbol: str, window: int = 30):
    """Get detailed volatility metrics for a stock"""
    return analyzer.calculate_volatility_metrics(symbol.upper(), window)

@app.get("/api/portfolio/risk")
async def get_portfolio_risk_analysis(symbols: str):
    """Get portfolio risk analysis for multiple symbols (comma-separated)"""
    symbol_list = [s.strip().upper() for s in symbols.split(",")]
    return analyzer.get_portfolio_risk(symbol_list)

@app.get("/api/market/overview")
async def get_market_overview():
    """Get market overview with major indices"""
    major_indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]  # S&P 500, Dow, NASDAQ, VIX
    market_data = {}
    
    for index in major_indices:
        try:
            data = analyzer.get_stock_data(index, "1mo")
            # Clean any NaN values before returning
            cleaned_data = {}
            for key, value in data.items():
                if isinstance(value, float) and pd.isna(value):
                    cleaned_data[key] = 0.0
                else:
                    cleaned_data[key] = value
            market_data[index] = cleaned_data
        except Exception as e:
            logger.error(f"Error fetching market data for {index}: {str(e)}")
            market_data[index] = {"error": "Data unavailable"}
    
    return market_data

@app.get("/api/stocks/{symbol}/chart")
async def get_stock_chart_data(symbol: str, period: str = "1y"):
    """Get chart data for plotting"""
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
        
        # Calculate volatility
        hist['Returns'] = hist['Close'].pct_change()
        hist['Volatility'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252)
        
        # Clean NaN values
        hist = hist.fillna(0)
        
        return {
            "dates": hist.index.strftime('%Y-%m-%d').tolist(),
            "prices": [float(x) for x in hist['Close'].round(2).tolist()],
            "volumes": [int(x) if not pd.isna(x) else 0 for x in hist['Volume'].tolist()],
            "volatility": [float(x) if not pd.isna(x) else 0.0 for x in hist['Volatility'].round(4).tolist()]
        }
    except Exception as e:
        logger.error(f"Error fetching chart data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching chart data for {symbol}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 