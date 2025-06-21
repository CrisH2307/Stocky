#!/usr/bin/env python3
"""
Debug script to test yfinance directly
"""

import yfinance as yf
import pandas as pd
import numpy as np

def test_yfinance():
    print("🧪 Testing yfinance directly...")
    
    try:
        # Test basic stock data
        print("Testing AAPL data...")
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1y")
        
        if hist.empty:
            print("❌ No data returned for AAPL")
            return False
        
        print(f"✅ AAPL data retrieved successfully!")
        print(f"   - Data points: {len(hist)}")
        print(f"   - Date range: {hist.index[0]} to {hist.index[-1]}")
        print(f"   - Current price: ${hist['Close'].iloc[-1]:.2f}")
        
        # Test volatility calculation
        print("\nTesting volatility calculation...")
        hist['Returns'] = hist['Close'].pct_change()
        hist['Volatility'] = hist['Returns'].rolling(window=30).std() * np.sqrt(252)
        
        current_vol = hist['Volatility'].iloc[-1]
        print(f"✅ Volatility calculated: {current_vol:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        return False

def test_market_data():
    print("\n🌍 Testing market indices...")
    
    indices = ["^GSPC", "^DJI", "^IXIC", "^VIX"]
    
    for index in indices:
        try:
            ticker = yf.Ticker(index)
            hist = ticker.history(period="1mo")
            
            if not hist.empty:
                print(f"✅ {index}: ${hist['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ {index}: No data")
                
        except Exception as e:
            print(f"❌ {index}: Error - {str(e)}")

if __name__ == "__main__":
    print("🔍 yfinance Debug Test")
    print("=" * 40)
    
    success = test_yfinance()
    test_market_data()
    
    if success:
        print("\n✅ yfinance is working correctly!")
        print("The issue might be in the API error handling.")
    else:
        print("\n❌ yfinance has issues. Check your internet connection.") 