import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import os
import datetime
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
import database

# Set page configuration
st.set_page_config(
    page_title="Electricity Price Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the database
database.initialize()

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
if 'dataset_id' not in st.session_state:
    st.session_state.dataset_id = None
if 'forecast_id' not in st.session_state:
    st.session_state.forecast_id = None
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Upload"

# Application header
st.title("⚡ Electricity Price Forecasting")
st.markdown("""
This application provides 30-day electricity price forecasts using advanced machine learning techniques.
Upload historical price data or use saved datasets.
""")

# Create navigation tabs
nav_tabs = ["Upload & Forecast", "Saved Datasets", "Saved Forecasts"]
nav_icons = ["📊", "💾", "📈"]

# Add navigation
cols = st.columns(len(nav_tabs))
for i, (tab, icon) in enumerate(zip(nav_tabs, nav_icons)):
    if cols[i].button(f"{icon} {tab}", key=f"nav_{i}", use_container_width=True):
        st.session_state.active_tab = tab

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
        
        # Data frequency selection
        frequency = st.radio(
            "Data Frequency",
            ["Hourly (default)", "5-Minute"],
            index=0,
            help="""Choose the frequency for data analysis:
            - Hourly: Standard resolution for long-term forecasting
            - 5-Minute: Higher resolution for more detailed short-term forecasting
            
            The original data will be automatically resampled to this frequency.
            If your original data is at a different frequency, the system will intelligently interpolate or aggregate as needed."""
        )
        
        # Convert user-friendly frequency to pandas format
        target_frequency = '5T' if frequency == "5-Minute" else 'H'
        
        # Model configuration
        st.subheader("LSTM Model Configuration")
        st.info("Using LSTM neural network with 3 layers, 750 hidden units, and 72 time steps.")
        
        # Information about the model configuration
        with st.expander("Model Details", expanded=False):
            st.markdown("""
            **LSTM Configuration:**
            - 3 LSTM layers with 750 hidden units each
            - Input sequence length: 72 time steps
            - Dropout rate: 0.3
            - Learning rate scheduler starting at 0.001
            - 100 training epochs
            """)
        
        # No model selection needed since we only use LSTM
        
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

# Main content area for Upload & Forecast tab
if st.session_state.active_tab == "Upload & Forecast":
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
                    
                    # Option to save dataset to database
                    with st.form(key="save_dataset_form"):
                        st.subheader("Save to Database")
                        dataset_name = st.text_input("Dataset Name", value=f"Dataset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                        dataset_description = st.text_area("Description (optional)", 
                                                         placeholder="Enter a description for this dataset...")
                        
                        save_submitted = st.form_submit_button("Save Dataset")
                        
                        if save_submitted:
                            try:
                                # Save to database
                                dataset_id = database.save_dataset(data, dataset_name, dataset_description)
                                st.session_state.dataset_id = dataset_id
                                st.success(f"Dataset saved to database with ID: {dataset_id}")
                            except Exception as e:
                                st.error(f"Error saving dataset to database: {str(e)}")
            else:
                st.error(f"Invalid data format: {validation_message}")
                st.session_state.data = None
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            st.session_state.data = None

    # Process data and generate forecast when the button is clicked
    if st.session_state.data is not None:
        st.write("Data is present in session state")
        if 'train_button' in locals():
            st.write("Train button variable exists in locals")
            if train_button:
                st.write("Train button is clicked!")
                with st.spinner("Processing data and training models..."):
                    try:
                        # Preprocess the data with selected frequency
                        # Get the target frequency from session state or use hourly as default
                        if 'frequency' in locals():
                            st.write(f"Frequency selected: {frequency}")
                            target_frequency = '5T' if frequency == "5-Minute" else 'H'
                        else:
                            st.write("No frequency in locals, defaulting to hourly")
                            target_frequency = 'H'
                        
                        st.write(f"Starting preprocessing with target frequency: {target_frequency}")
                        processed_data = preprocess_data(st.session_state.data, target_frequency)
                        st.session_state.processed_data = processed_data
                        st.write(f"Preprocessing complete! Shape: {processed_data.shape}")
                        
                        # Verify timestamp is in datetime format
                        if not pd.api.types.is_datetime64_any_dtype(processed_data['timestamp']):
                            processed_data['timestamp'] = pd.to_datetime(processed_data['timestamp'])
                            st.write("Converted timestamp to datetime format")
                
                        # Engineer features
                        st.write("Starting feature engineering...")
                        features, decomposition = engineer_features(processed_data)
                        st.write(f"Feature engineering generated {features.shape[1]} features with {features.shape[0]} rows")
                        
                        # Identify and handle any NaN values
                        nan_count = features.isna().sum().sum()
                        if nan_count > 0:
                            st.write(f"Found {nan_count} NaN values in features. Cleaning up...")
                            features = features.fillna(0)
                            
                        # Ensure price column from processed_data is aligned with features
                        st.write(f"Feature index length: {len(features.index)}")
                        st.write(f"Price series length: {len(processed_data['price'])}")
                        
                        # Make sure we're using the processed price data that matches our features
                        price_data = processed_data.loc[processed_data['timestamp'].isin(features['timestamp']), 'price']
                        st.write(f"Aligned price data length: {len(price_data)}")
                        
                        st.session_state.features = features
                        st.session_state.decomposition = decomposition
                        st.write("Feature engineering complete!")
                        
                        # Initialize and train the forecaster
                        st.write("Initializing LSTM forecaster")
                        forecaster = ElectricityPriceForecaster(
                            forecast_horizon=forecast_days,
                            confidence_interval=conf_interval
                        )
                        
                        # Train the models and generate forecast
                        st.write("Starting model training...")
                        forecast, error_metrics, feature_importance = forecaster.train_and_forecast(
                            features, price_data
                        )
                        
                        # Store the results in session state
                        st.write("Saving results to session state...")
                        st.session_state.forecast = forecast
                        st.session_state.error_metrics = error_metrics
                        st.session_state.feature_importance = feature_importance
                        st.session_state.models = forecaster
                        
                        st.success("Models trained and forecast generated successfully!")
                        
                    except Exception as e:
                        st.error(f"An error occurred during model training and forecasting: {str(e)}")
                
                # Option to save forecast to database if a dataset_id is available
                if st.session_state.forecast is not None and st.session_state.dataset_id is not None:
                    with st.form(key="save_forecast_form"):
                        st.subheader("Save Forecast to Database")
                        
                        forecast_name = st.text_input("Forecast Name", 
                                                   value=f"Forecast_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
                        
                        save_forecast_submitted = st.form_submit_button("Save Forecast")
                        
                        if save_forecast_submitted:
                            try:
                                # Use only LSTM model for the database entry
                                models_list = ["LSTM"]
                                
                                # Save forecast to database
                                forecast_id = database.save_forecast(
                                    dataset_id=st.session_state.dataset_id,
                                    forecast_df=st.session_state.forecast,
                                    error_metrics=st.session_state.error_metrics,
                                    feature_importance=st.session_state.feature_importance,
                                    name=forecast_name,
                                    forecast_horizon=forecast_days,
                                    confidence_interval=conf_interval,
                                    models_used=models_list
                                )
                                
                                st.session_state.forecast_id = forecast_id
                                st.success(f"Forecast saved to database with ID: {forecast_id}")
                            except Exception as e:
                                st.error(f"Error saving forecast to database: {str(e)}")
                elif st.session_state.forecast is not None:
                    st.info("Save the dataset to the database first to enable forecast saving.")

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

# Different content based on the active tab
if st.session_state.active_tab == "Saved Datasets":
    st.header("💾 Saved Datasets")
    
    # Get all datasets from database
    try:
        datasets = database.get_all_datasets()
        
        if not datasets:
            st.info("No datasets found in the database. Upload a dataset first.")
        else:
            # Display datasets in a table
            st.write(f"Found {len(datasets)} datasets in the database.")
            
            # Create a DataFrame for display
            dataset_data = []
            for ds in datasets:
                dataset_data.append({
                    "ID": ds.id,
                    "Name": ds.name,
                    "Description": ds.description,
                    "Records": ds.record_count,
                    "Created": ds.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Time Range": f"{ds.start_date.strftime('%Y-%m-%d')} to {ds.end_date.strftime('%Y-%m-%d')}",
                    "External Features": "Yes" if ds.has_external_features else "No"
                })
            
            dataset_df = pd.DataFrame(dataset_data)
            st.dataframe(dataset_df, use_container_width=True)
            
            # Select dataset to load
            selected_dataset_id = st.selectbox(
                "Select a dataset to load",
                options=[ds.id for ds in datasets],
                format_func=lambda x: f"ID: {x} - {next((ds.name for ds in datasets if ds.id == x), 'Unknown')}"
            )
            
            if st.button("Load Selected Dataset"):
                with st.spinner("Loading dataset..."):
                    try:
                        # Load dataset from database
                        data = database.get_dataset_data(selected_dataset_id)
                        
                        # Set session state
                        st.session_state.data = data
                        st.session_state.dataset_id = selected_dataset_id
                        
                        # Reset other session variables
                        st.session_state.processed_data = None
                        st.session_state.features = None
                        st.session_state.forecast = None
                        st.session_state.error_metrics = None
                        st.session_state.feature_importance = None
                        st.session_state.decomposition = None
                        st.session_state.models = None
                        
                        # Change tab
                        st.session_state.active_tab = "Upload & Forecast"
                        
                        st.success(f"Dataset loaded successfully! Switched to Upload & Forecast tab.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading dataset: {str(e)}")
            
            # Option to delete a dataset
            if st.checkbox("Delete Dataset"):
                if st.button("Delete Selected Dataset", type="primary", help="This action cannot be undone!"):
                    try:
                        success = database.delete_dataset(selected_dataset_id)
                        if success:
                            st.success(f"Dataset {selected_dataset_id} deleted successfully!")
                            # If the deleted dataset was the active one, reset the session state
                            if st.session_state.dataset_id == selected_dataset_id:
                                st.session_state.data = None
                                st.session_state.dataset_id = None
                            st.rerun()
                        else:
                            st.error("Failed to delete dataset.")
                    except Exception as e:
                        st.error(f"Error deleting dataset: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error loading datasets from database: {str(e)}")

elif st.session_state.active_tab == "Saved Forecasts":
    st.header("📈 Saved Forecasts")
    
    # Get all forecasts from database
    try:
        forecasts = database.get_all_forecasts()
        
        if not forecasts:
            st.info("No forecasts found in the database. Create a forecast first.")
        else:
            # Display forecasts in a table
            st.write(f"Found {len(forecasts)} forecasts in the database.")
            
            # Create a DataFrame for display
            forecast_data = []
            for f in forecasts:
                forecast_data.append({
                    "ID": f.id,
                    "Name": f.name,
                    "Dataset ID": f.dataset_id,
                    "Created": f.created_at.strftime("%Y-%m-%d %H:%M"),
                    "Horizon (days)": f.forecast_horizon,
                    "Confidence Interval": f"{f.confidence_interval}%",
                    "Models": f.models_used
                })
            
            forecast_df = pd.DataFrame(forecast_data)
            st.dataframe(forecast_df, use_container_width=True)
            
            # Select forecast to load
            selected_forecast_id = st.selectbox(
                "Select a forecast to load",
                options=[f.id for f in forecasts],
                format_func=lambda x: f"ID: {x} - {next((f.name for f in forecasts if f.id == x), 'Unknown')}"
            )
            
            if st.button("Load Selected Forecast"):
                with st.spinner("Loading forecast..."):
                    try:
                        # Load forecast from database
                        forecast_df, error_metrics, feature_importance = database.get_forecast_data(selected_forecast_id)
                        
                        # Get the associated dataset
                        selected_forecast = next((f for f in forecasts if f.id == selected_forecast_id), None)
                        if selected_forecast:
                            data = database.get_dataset_data(selected_forecast.dataset_id)
                            # Determine which frequency was used in the original forecast
                            # For now, just default to hourly since we don't store frequency in the database
                            processed_data = preprocess_data(data, 'H')
                            
                            # Set session state
                            st.session_state.data = data
                            st.session_state.dataset_id = selected_forecast.dataset_id
                            st.session_state.processed_data = processed_data
                            st.session_state.forecast = forecast_df
                            st.session_state.error_metrics = error_metrics
                            st.session_state.feature_importance = feature_importance
                            st.session_state.forecast_id = selected_forecast_id
                            
                            # Initialize a forecaster object with the loaded parameters
                            forecaster = ElectricityPriceForecaster(
                                forecast_horizon=selected_forecast.forecast_horizon,
                                confidence_interval=selected_forecast.confidence_interval
                            )
                            st.session_state.models = forecaster
                            
                            # Change tab
                            st.session_state.active_tab = "Upload & Forecast"
                            
                            st.success(f"Forecast loaded successfully! Switched to Upload & Forecast tab.")
                            st.rerun()
                        else:
                            st.error("Could not find the selected forecast.")
                    except Exception as e:
                        st.error(f"Error loading forecast: {str(e)}")
            
            # Option to delete a forecast
            if st.checkbox("Delete Forecast"):
                if st.button("Delete Selected Forecast", type="primary", help="This action cannot be undone!"):
                    try:
                        success = database.delete_forecast(selected_forecast_id)
                        if success:
                            st.success(f"Forecast {selected_forecast_id} deleted successfully!")
                            # If the deleted forecast was the active one, reset the session state
                            if st.session_state.forecast_id == selected_forecast_id:
                                st.session_state.forecast = None
                                st.session_state.error_metrics = None
                                st.session_state.feature_importance = None
                                st.session_state.forecast_id = None
                            st.rerun()
                        else:
                            st.error("Failed to delete forecast.")
                    except Exception as e:
                        st.error(f"Error deleting forecast: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error loading forecasts from database: {str(e)}")

# Display welcome message if needed
elif st.session_state.active_tab == "Upload & Forecast" and st.session_state.data is None:
    st.info("""
    ### Getting Started
    1. Upload your historical electricity price data (CSV or JSON format)
    2. Configure the forecast parameters in the sidebar
    3. Click on "Train Models & Generate Forecast" to see the results
    
    For optimal forecasting performance, provide at least 1 year of data.
    
    **New Feature**: The application now supports both hourly and 5-minute frequency data!
    
    You can also load previously saved datasets from the "Saved Datasets" tab.
    """)

# Footer
st.markdown("---")
st.markdown("""
**Electricity Price Forecasting Application** | Powered by advanced machine learning algorithms | ⚡
""")
