# Electricity Price Forecasting Application

This application provides 30-day electricity price forecasts using advanced machine learning techniques including LSTM neural networks. The application is optimized to run on various platforms including M1 MacBook Air.

## Features

- Upload and validate electricity price data
- Generate forecasts with configurable parameters
- Visualize forecast results with interactive charts
- Save and retrieve datasets and forecasts
- Optimized LSTM model with early stopping and mixed precision training

## Local Setup

### Option 1: Using setup script

1. Clone this repository
2. Make the setup script executable:
   ```
   chmod +x setup.sh
   ```
3. Run the setup script:
   ```
   ./setup.sh
   ```

### Option 2: Manual setup

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up the environment variable:
   ```
   export DATABASE_URL=sqlite:///electricity_forecasting.db
   ```
   Or create a `.env` file in the project root with:
   ```
   DATABASE_URL=sqlite:///electricity_forecasting.db
   ```
4. Initialize the database:
   ```
   python init_db.py
   ```
5. Run the application:
   ```
   streamlit run app.py
   ```

## Data Format

The application expects CSV or JSON data with the following columns:
- `timestamp`: Date and time in ISO format or m/d/yy h:mm format
- `price`: Electricity price value

Optional columns:
- `temperature`: Temperature value
- `wind_speed`: Wind speed value
- `solar_irradiance`: Solar irradiance value
- `precipitation`: Precipitation value
- `load`: System load value

## Model Details

The forecasting model uses:
- LSTM neural network with 3 layers and 750 hidden units
- 72 time step sequence length
- Early stopping for efficient training
- Mixed precision training for optimized performance on M1 Macs
- Metal Performance Shaders (MPS) optimization for Apple Silicon

## Deployment

For production deployment, consider:
- Using PostgreSQL instead of SQLite
- Setting up a proper authentication system
- Deploying to a cloud provider (AWS, GCP, Azure)
- Using Streamlit Cloud for easy hosting 