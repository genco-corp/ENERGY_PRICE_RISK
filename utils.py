import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime, timedelta

def download_csv(df):
    """
    Generate a CSV download from a DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame to convert to CSV
        
    Returns:
        str: CSV string for download
    """
    # Create a copy of the dataframe to avoid modifying the original
    download_df = df.copy()
    
    # Convert timestamp to string if it's a datetime object
    if 'timestamp' in download_df.columns and pd.api.types.is_datetime64_any_dtype(download_df['timestamp']):
        download_df['timestamp'] = download_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Convert DataFrame to CSV
    csv_buffer = io.StringIO()
    download_df.to_csv(csv_buffer, index=False)
    csv_str = csv_buffer.getvalue()
    
    return csv_str

def create_forecast_dates(start_date, forecast_days=30):
    """
    Create a series of dates for forecasting
    
    Args:
        start_date (datetime): Starting date for forecast
        forecast_days (int): Number of days to forecast
        
    Returns:
        list: List of datetime objects for forecast period
    """
    forecast_hours = forecast_days * 24
    dates = [start_date + timedelta(hours=h) for h in range(forecast_hours)]
    return dates

def format_timestamp(timestamp):
    """
    Format timestamp for display
    
    Args:
        timestamp (datetime): Timestamp to format
        
    Returns:
        str: Formatted timestamp string
    """
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
        
    return timestamp.strftime('%Y-%m-%d %H:%M')

def calculate_percentage_change(current, previous):
    """
    Calculate percentage change between two values
    
    Args:
        current (float): Current value
        previous (float): Previous value
        
    Returns:
        float: Percentage change
    """
    if previous == 0:
        return 0
    
    return ((current - previous) / previous) * 100

def detect_price_spikes(forecast_df, threshold=3):
    """
    Detect potential price spikes in forecast
    
    Args:
        forecast_df (pd.DataFrame): Forecast dataframe
        threshold (float): Threshold for spike detection (standard deviations)
        
    Returns:
        pd.Series: Boolean series indicating spikes
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = forecast_df['predicted'].rolling(window=24, min_periods=1).mean()
    rolling_std = forecast_df['predicted'].rolling(window=24, min_periods=1).std()
    
    # Define spikes as values that are significantly higher than the rolling mean
    spikes = forecast_df['predicted'] > (rolling_mean + threshold * rolling_std)
    
    return spikes

def detect_price_drops(forecast_df, threshold=3):
    """
    Detect potential price drops in forecast
    
    Args:
        forecast_df (pd.DataFrame): Forecast dataframe
        threshold (float): Threshold for drop detection (standard deviations)
        
    Returns:
        pd.Series: Boolean series indicating drops
    """
    # Calculate rolling mean and standard deviation
    rolling_mean = forecast_df['predicted'].rolling(window=24, min_periods=1).mean()
    rolling_std = forecast_df['predicted'].rolling(window=24, min_periods=1).std()
    
    # Define drops as values that are significantly lower than the rolling mean
    drops = forecast_df['predicted'] < (rolling_mean - threshold * rolling_std)
    
    return drops

def aggregate_by_day(forecast_df):
    """
    Aggregate hourly forecast to daily values
    
    Args:
        forecast_df (pd.DataFrame): Hourly forecast dataframe
        
    Returns:
        pd.DataFrame: Daily aggregated forecast
    """
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(forecast_df['timestamp']):
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    
    # Create a date column for grouping
    forecast_df['date'] = forecast_df['timestamp'].dt.date
    
    # Group by date and aggregate
    daily_df = forecast_df.groupby('date').agg({
        'predicted': 'mean',
        'actual': 'mean'
    }).reset_index()
    
    # If confidence intervals exist, aggregate them too
    if 'lower' in forecast_df.columns and 'upper' in forecast_df.columns:
        ci_df = forecast_df.groupby('date').agg({
            'lower': 'mean',
            'upper': 'mean'
        })
        daily_df = daily_df.merge(ci_df, on='date')
    
    # Convert date back to datetime for plotting
    daily_df['timestamp'] = pd.to_datetime(daily_df['date'])
    
    return daily_df
