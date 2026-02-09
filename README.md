# service-desk-analytics
Live Streamlit dashboard visualizing service desk efficiency metrics using DuckDB.

## Features

### Dashboard
- Live Streamlit dashboard visualizing service desk efficiency metrics
- DuckDB-powered analytics for fast querying of operational data
- Time-indexed metrics including tickets over time, SLA snapshots, MTTR, and backlog growth

### Forecasting
- **SARIMAX modeling** for ticket volume and backlog trend analysis
- **Prophet** for seasonality and trend decomposition
- **LSTM** neural network benchmark for comparison
- Model evaluation using MAE and RMSE metrics
- Forecast vs. actual visualizations for 4–8 week planning horizons
- [View the forecasting notebook on Kaggle](https://www.kaggle.com/code/doctorn8/time-series-service-desk-analytics)

## Project Structure

```
service-desk-analytics/
├── app.py                      # Streamlit dashboard
├── service_desk.duckdb         # DuckDB database with operational data
├── requirements.txt            # Python dependencies
└── notebooks/                  # Analytics notebooks
    └── README.md              # Links to Kaggle forecasting notebook
```

## Installation

```bash
# Clone the repository
git clone https://github.com/DoctorN8/service-desk-analytics.git
cd service-desk-analytics

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run app.py
```

## Tech Stack
- **Python**: Core language
- **Streamlit**: Interactive dashboard
- **DuckDB**: Embedded analytical database
- **statsmodels**: SARIMAX time series modeling
- **Prophet**: Facebook's forecasting library
- **TensorFlow/Keras**: LSTM neural networks
- **pandas**: Data manipulation
- **matplotlib/plotly**: Visualizations
