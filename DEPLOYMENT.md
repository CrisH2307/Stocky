# 🚀 Deployment Guide - Stock Volatility Dashboard

## Quick Deploy Options for Hackathon

### Option 1: Local Development (Recommended for Demo)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the application
python start.py
```

**Access URLs:**

- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Streamlit Cloud (Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Deploy automatically

### Option 3: Railway (Free Tier)

1. Install Railway CLI: `npm i -g @railway/cli`
2. Create `railway.json`:

```json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### Option 4: Render (Free Tier)

1. Create `render.yaml`:

```yaml
services:
  - type: web
    name: stock-volatility-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Option 5: Heroku (Free Tier Discontinued)

1. Create `Procfile`:

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

## Environment Variables

Create `.env` file:

```env
PORT=8000
ENVIRONMENT=production
```

## Production Considerations

### Security

- Add rate limiting
- Implement API key authentication
- Add CORS restrictions
- Use HTTPS

### Performance

- Add Redis caching
- Implement database connection pooling
- Add request/response compression

### Monitoring

- Add logging
- Implement health checks
- Add metrics collection

## Demo Preparation

### 1. Backup Plan

- Have screenshots ready
- Prepare mock data
- Test with different stocks

### 2. Demo Script

```
1. Show market overview
2. Analyze a volatile stock (TSLA)
3. Show portfolio risk analysis
4. Demonstrate real-time updates
5. Show API documentation
```

### 3. Key Features to Highlight

- Real-time data fetching
- Volatility calculations
- Interactive charts
- Responsive design
- API-first architecture

## Troubleshooting

### Common Issues

1. **Port already in use**: Change port in `main.py`
2. **CORS errors**: Check CORS middleware configuration
3. **Data not loading**: Verify yfinance is working
4. **Charts not rendering**: Check Chart.js/Plotly dependencies

### Debug Mode

```bash
# Run with debug logging
uvicorn main:app --reload --log-level debug
```

## Performance Tips

### For Demo

- Use cached data for initial load
- Implement progressive loading
- Add loading states
- Optimize chart rendering

### For Production

- Add database for data persistence
- Implement proper caching strategy
- Use CDN for static assets
- Add monitoring and alerting

## API Endpoints Summary

| Endpoint                     | Method | Description        |
| ---------------------------- | ------ | ------------------ |
| `/`                          | GET    | Health check       |
| `/api/stocks/{symbol}`       | GET    | Stock data         |
| `/api/volatility/{symbol}`   | GET    | Volatility metrics |
| `/api/portfolio/risk`        | GET    | Portfolio analysis |
| `/api/market/overview`       | GET    | Market indices     |
| `/api/stocks/{symbol}/chart` | GET    | Chart data         |

## Frontend Options

### 1. Streamlit (Current)

- Pros: Fast development, built-in components
- Cons: Limited customization

### 2. React + Tailwind (HTML Template)

- Pros: Full customization, modern UI
- Cons: More development time

### 3. Vue.js + Vuetify

- Pros: Easy to learn, material design
- Cons: Additional framework

## Success Metrics

- [ ] API responds within 2 seconds
- [ ] Charts render smoothly
- [ ] Real-time updates work
- [ ] Mobile responsive
- [ ] Error handling works
- [ ] Demo runs without issues

Good luck with your hackathon! 🚀
