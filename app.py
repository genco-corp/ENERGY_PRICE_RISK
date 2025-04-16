import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import os
import datetime
from datetime import timedelta

from data_processing import preprocess_data, validate_data
from feature_engineering import engineer_features
from models import ElectricityPriceForecaster
from visualization import (
    plot_price_forecast,
    plot_feature_importance,
    plot_error_metrics,
    plot_decomposition,
    plot_historical_prices
)
from utils import download_csv

# Set page configuration
st.set_page_config(
    page_title="Electricity Price Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'features' not in st.session_state:
    st.session_state.features = None
if 'forecast' not in st.session_state:
    st.session_state.forecast = None
if 'error_metrics' not in st.session_state:
    st.session_state.error_metrics = None
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'decomposition' not in st.session_state:
    st.session_state.decomposition = None
if 'models' not in st.session_state:
    st.session_state.models = None

# Application header
st.title("⚡ Electricity Price Forecasting")
st.markdown("""
This application provides 30-day electricity price forecasts using advanced machine learning techniques.
Upload your historical price data to get started.
""")

# Sidebar for inputs and configuration
with st.sidebar:
    st.header("Configuration")
    
    # Data upload section
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader(
        "Upload historical electricity price data (CSV or JSON)",
        type=["csv", "json"]
    )
    
    # Example data format
    with st.expander("Expected Data Format"):
        st.markdown("""
        Your data should have the following columns:
        - **timestamp**: Date and time in one of these formats:
          - ISO format (YYYY-MM-DD HH:MM:SS)
          - m/d/yy h:mm (e.g., 1/24/24 0:05)
        - **price**: Electricity price value
        
        Optional columns:
        - **temperature**: Temperature value
        - **wind_speed**: Wind speed value
        - **solar_irradiance**: Solar irradiance value
        - **precipitation**: Precipitation value
        - **load**: System load value
        
        Examples:
        ```
        # ISO format example:
        timestamp,price,temperature
        2022-01-01 00:00:00,45.2,12.5
        2022-01-01 01:00:00,42.8,12.1
        
        # m/d/yy h:mm format example:
        timestamp,price
        1/24/24 0:05,42.5
        1/24/24 0:10,43.2
        ...
        ```
        """)
    
    # Only show further options if data is uploaded
    if uploaded_file is not None or st.session_state.data is not None:
        # Forecast parameters
        st.subheader("Forecast Parameters")
        forecast_days = st.slider("Forecast Horizon (days)", 1, 30, 7)
        
        # Model selection
        st.subheader("Model Configuration")
        models_to_use = st.multiselect(
            "Select Models",
            ["LSTM", "XGBoost", "Prophet", "Ensemble (All)"],
            default=["Ensemble (All)"]
        )
        
        # If no specific model is selected, default to ensemble
        if not models_to_use:
            models_to_use = ["Ensemble (All)"]
        
        # Confidence interval settings
        conf_interval = st.select_slider(
            "Confidence Interval",
            options=[80, 90, 95],
            value=90
        )
        
        # Training button
        train_button = st.button("Train Models & Generate Forecast")
        
        # Download results button (only shown if forecast exists)
        if st.session_state.forecast is not None:
            st.subheader("Export Results")
            st.download_button(
                label="Download Forecast as CSV",
                data=download_csv(st.session_state.forecast),
                file_name="electricity_price_forecast.csv",
                mime="text/csv"
            )

# Main content area
if uploaded_file is not None:
    try:
        # Process the uploaded file
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            data = pd.read_json(uploaded_file)
        
        # Validate the data
        validation_result, validation_message = validate_data(data)
        
        if validation_result:
            st.session_state.data = data
            st.success("Data uploaded successfully!")
            
            # Display dataset info
            with st.expander("Dataset Overview", expanded=True):
                st.write(f"**Number of records:** {len(data)}")
                st.write(f"**Time range:** {data['timestamp'].min()} to {data['timestamp'].max()}")
                st.write(f"**Columns:** {', '.join(data.columns.tolist())}")
                st.write("**Sample data:**")
                st.dataframe(data.head())
                
                # Show historical price chart
                if 'price' in data.columns:
                    st.plotly_chart(plot_historical_prices(data), use_container_width=True)
        else:
            st.error(f"Invalid data format: {validation_message}")
            st.session_state.data = None
    except Exception as e:
        st.error(f"Error processing the file: {str(e)}")
        st.session_state.data = None

# Process data and generate forecast when the button is clicked
if st.session_state.data is not None and 'train_button' in locals() and train_button:
    with st.spinner("Processing data and training models..."):
        try:
            # Preprocess the data
            processed_data = preprocess_data(st.session_state.data)
            st.session_state.processed_data = processed_data
            
            # Engineer features
            features, decomposition = engineer_features(processed_data)
            st.session_state.features = features
            st.session_state.decomposition = decomposition
            
            # Initialize and train the forecaster
            forecaster = ElectricityPriceForecaster(
                models_to_use=models_to_use,
                forecast_horizon=forecast_days,
                confidence_interval=conf_interval
            )
            
            # Train the models and generate forecast
            forecast, error_metrics, feature_importance = forecaster.train_and_forecast(
                features, processed_data['price']
            )
            
            # Store the results in session state
            st.session_state.forecast = forecast
            st.session_state.error_metrics = error_metrics
            st.session_state.feature_importance = feature_importance
            st.session_state.models = forecaster
            
            st.success("Models trained and forecast generated successfully!")
        except Exception as e:
            st.error(f"An error occurred during model training and forecasting: {str(e)}")

# Display forecast results if available
if st.session_state.forecast is not None:
    st.header("Forecast Results")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "Price Forecast", 
        "Model Performance", 
        "Feature Importance", 
        "Time Series Decomposition"
    ])
    
    with tab1:
        st.subheader("30-Day Price Forecast")
        st.plotly_chart(
            plot_price_forecast(
                st.session_state.forecast, 
                st.session_state.processed_data,
                confidence_interval=st.session_state.models.confidence_interval
            ),
            use_container_width=True
        )
        
        # Display forecast data table
        with st.expander("Forecast Data Table"):
            st.dataframe(st.session_state.forecast)
    
    with tab2:
        st.subheader("Model Performance Metrics")
        st.plotly_chart(
            plot_error_metrics(st.session_state.error_metrics),
            use_container_width=True
        )
        
        # Display metrics in table format
        with st.expander("Detailed Metrics"):
            st.dataframe(pd.DataFrame(st.session_state.error_metrics))
    
    with tab3:
        st.subheader("Feature Importance Analysis")
        if st.session_state.feature_importance is not None:
            st.plotly_chart(
                plot_feature_importance(st.session_state.feature_importance),
                use_container_width=True
            )
        else:
            st.info("Feature importance analysis is not available for the selected model(s).")
    
    with tab4:
        st.subheader("Time Series Decomposition")
        if st.session_state.decomposition is not None:
            st.plotly_chart(
                plot_decomposition(st.session_state.decomposition),
                use_container_width=True
            )
        else:
            st.info("Time series decomposition is not available.")

# Initial guidance for new users
if st.session_state.data is None:
    st.info("""
    ### Getting Started
    1. Upload your historical electricity price data (CSV or JSON format)
    2. Configure the forecast parameters in the sidebar
    3. Click on "Train Models & Generate Forecast" to see the results
    
    For optimal forecasting performance, provide at least 1 year of hourly data.
    """)

# Footer
st.markdown("---")
st.markdown("""
**Electricity Price Forecasting Application** | Powered by advanced machine learning algorithms | ⚡
""")
