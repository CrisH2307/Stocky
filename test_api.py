#!/usr/bin/env python3
"""
Test script for Stock Volatility Dashboard API
Run this to verify all endpoints work correctly
"""

import requests
import json
import time

API_BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, description):
    """Test an API endpoint"""
    print(f"Testing {description}...")
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=10)
        if response.status_code == 200:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED (Status: {response.status_code})")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def test_stock_data():
    """Test stock data endpoints"""
    print("\n📊 Testing Stock Data Endpoints")
    print("=" * 40)
    
    # Test basic stock data
    success1 = test_endpoint("/api/stocks/AAPL", "AAPL Stock Data")
    
    # Test volatility data
    success2 = test_endpoint("/api/volatility/AAPL", "AAPL Volatility")
    
    # Test chart data
    success3 = test_endpoint("/api/stocks/AAPL/chart", "AAPL Chart Data")
    
    return success1 and success2 and success3

def test_market_data():
    """Test market overview endpoint"""
    print("\n🌍 Testing Market Data Endpoints")
    print("=" * 40)
    
    success = test_endpoint("/api/market/overview", "Market Overview")
    return success

def test_portfolio_data():
    """Test portfolio analysis endpoint"""
    print("\n💼 Testing Portfolio Analysis Endpoints")
    print("=" * 40)
    
    success = test_endpoint("/api/portfolio/risk?symbols=AAPL,GOOGL,MSFT", "Portfolio Risk Analysis")
    return success

def test_health_check():
    """Test health check endpoint"""
    print("\n🏥 Testing Health Check")
    print("=" * 40)
    
    success = test_endpoint("/", "Health Check")
    return success

def test_api_docs():
    """Test API documentation"""
    print("\n📚 Testing API Documentation")
    print("=" * 40)
    
    success = test_endpoint("/docs", "API Documentation")
    return success

def main():
    print("🧪 Stock Volatility Dashboard API Test")
    print("=" * 50)
    
    # Wait for API to be ready
    print("Waiting for API to start...")
    time.sleep(2)
    
    # Test all endpoints
    tests = [
        test_health_check,
        test_stock_data,
        test_market_data,
        test_portfolio_data,
        test_api_docs
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! Your API is working correctly.")
        print("\n🚀 You can now:")
        print("   - Open http://localhost:8000/docs for API documentation")
        print("   - Run 'streamlit run dashboard.py' for the frontend")
        print("   - Use the HTML template in templates/index.html")
    else:
        print("⚠️  Some tests failed. Check your API server.")
        print("\n💡 Make sure to:")
        print("   - Run 'uvicorn main:app --reload' to start the backend")
        print("   - Check that all dependencies are installed")
        print("   - Verify the API is running on port 8000")

if __name__ == "__main__":
    main() 