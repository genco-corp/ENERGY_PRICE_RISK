import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def validate_data(data):
    """
    Validates the input data for required format and columns.
    
    Args:
        data (pd.DataFrame): Input data to validate
        
    Returns:
        tuple: (is_valid, message) indicating validation status and message
    """
    # Check if dataframe is empty
    if data.empty:
        return False, "The uploaded data is empty."
    
    # Check for required columns
    required_columns = ['timestamp', 'price']
    for col in required_columns:
        if col not in data.columns:
            return False, f"Missing required column: {col}"
    
    # Check timestamp format
    try:
        # Convert timestamp column to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
            # Try parsing with multiple formats including "m/d/yy h:mm"
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            except:
                try:
                    # Try with format m/d/yy h:mm
                    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%m/%d/%y %H:%M')
                except Exception as e2:
                    return False, f"Error converting timestamp with format 'm/d/yy h:mm': {str(e2)}"
    except Exception as e:
        return False, f"Error converting timestamp column to datetime: {str(e)}"
    
    # Check price column data type
    try:
        data['price'] = pd.to_numeric(data['price'])
    except Exception as e:
        return False, f"Error converting price column to numeric: {str(e)}"
    
    # Check if there's enough data (at least 30 days)
    min_date = data['timestamp'].min()
    max_date = data['timestamp'].max()
    days_span = (max_date - min_date).days
    
    if days_span < 30:
        return False, f"Data spans only {days_span} days. At least 30 days of data is recommended."
    
    return True, "Data validation successful."

def preprocess_data(data, target_frequency='H'):
    """
    Preprocesses the input data for forecasting.
    
    Args:
        data (pd.DataFrame): Input data with at least 'timestamp' and 'price' columns
        target_frequency (str): Target frequency for resampling ('H' for hourly, '5T' for 5-minute)
        
    Returns:
        pd.DataFrame: Preprocessed data ready for feature engineering
    """
    # Make a copy to avoid modifying the original data
    df = data.copy()
    
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        except:
            try:
                # Try with format m/d/yy h:mm
                df['timestamp'] = pd.to_datetime(df['timestamp'], format='%m/%d/%y %H:%M')
            except Exception as e:
                print(f"Error converting timestamp format: {str(e)}")
                # Fall back to trying force parse as last resort
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Set timestamp as index
    df.set_index('timestamp', inplace=True)
    
    # Sort by timestamp
    df.sort_index(inplace=True)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Resample to target frequency if needed
    current_freq = pd.infer_freq(df.index)
    if current_freq != target_frequency:
        df = resample_to_frequency(df, target_frequency)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Reset index to have timestamp as a column again
    df.reset_index(inplace=True)
    
    return df

def handle_missing_values(df):
    """
    Handles missing values in the dataset.
    
    Args:
        df (pd.DataFrame): Input dataframe with timestamp index
        
    Returns:
        pd.DataFrame: Dataframe with handled missing values
    """
    # Check for missing values
    if df['price'].isna().sum() > 0:
        # Interpolate missing prices
        df['price'] = df['price'].interpolate(method='time')
        
        # If there are still NAs at the beginning or end, use forward/backward fill
        df['price'] = df['price'].fillna(method='ffill')
        df['price'] = df['price'].fillna(method='bfill')
    
    # Handle other columns if they exist
    optional_columns = ['temperature', 'wind_speed', 'solar_irradiance', 
                        'precipitation', 'load']
    
    for col in optional_columns:
        if col in df.columns and df[col].isna().sum() > 0:
            df[col] = df[col].interpolate(method='time')
            df[col] = df[col].fillna(method='ffill')
            df[col] = df[col].fillna(method='bfill')
    
    return df

def resample_to_frequency(df, target_freq='H'):
    """
    Resamples data to the target frequency.
    
    Args:
        df (pd.DataFrame): Input dataframe with timestamp index
        target_freq (str): Target frequency ('H' for hourly, '5T' for 5 minutes)
        
    Returns:
        pd.DataFrame: Dataframe with the target frequency
    """
    # Make a copy and ensure index is datetime
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Sort index to ensure proper resampling
    df = df.sort_index()
    
    # Determine current frequency
    inferred_freq = pd.infer_freq(df.index)
    
    # If target frequency is hourly
    if target_freq == 'H':
        # If frequency is finer than hourly (e.g., 15min or 5min), resample to hourly
        if inferred_freq is not None and inferred_freq[-1] not in ['H', 'D']:
            df = df.resample('H').mean()
        # If frequency is larger than hourly (e.g., daily), interpolate to hourly
        elif inferred_freq is not None and inferred_freq[-1] == 'D':
            df = df.resample('H').interpolate(method='time')
    
    # If target frequency is 5-minute
    elif target_freq == '5T':
        # If frequency is coarser than 5min (e.g., hourly), interpolate to 5min
        if inferred_freq is not None and (inferred_freq[-1] in ['H', 'D'] or
                                        (inferred_freq[-1] == 'T' and int(inferred_freq[:-1]) > 5)):
            df = df.resample('5T').interpolate(method='time')
        # If frequency is finer than 5min, resample to 5min
        elif inferred_freq is not None and inferred_freq[-1] == 'T' and int(inferred_freq[:-1]) < 5:
            df = df.resample('5T').mean()
    
    # If current frequency couldn't be determined, force to target frequency
    elif inferred_freq is None:
        # First check the average time difference
        avg_diff = (df.index[-1] - df.index[0]) / len(df)
        
        if target_freq == 'H' and avg_diff < pd.Timedelta(hours=1):
            # Data appears to be sub-hourly, resample to hourly
            df = df.resample('H').mean()
        elif target_freq == '5T' and avg_diff < pd.Timedelta(minutes=5):
            # Data appears to be sub-5-minute, resample to 5-minute
            df = df.resample('5T').mean()
        elif target_freq == 'H' and avg_diff > pd.Timedelta(hours=1):
            # Data appears to be coarser than hourly, interpolate to hourly
            df = df.resample('H').interpolate(method='time')
        elif target_freq == '5T' and avg_diff > pd.Timedelta(minutes=5):
            # Data appears to be coarser than 5-minute, interpolate to 5-minute
            df = df.resample('5T').interpolate(method='time')
    
    return df

# Keep the original function for backward compatibility
def resample_to_hourly(df):
    """
    Resamples data to hourly frequency if needed (legacy function).
    
    Args:
        df (pd.DataFrame): Input dataframe with timestamp index
        
    Returns:
        pd.DataFrame: Dataframe with hourly frequency
    """
    return resample_to_frequency(df, 'H')

def handle_outliers(df):
    """
    Detects and handles outliers in the price data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with handled outliers
    """
    # Calculate IQR for price
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define outlier bounds (using 3*IQR for less aggressive filtering)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Create a mask for outliers
    outliers = (df['price'] < lower_bound) | (df['price'] > upper_bound)
    
    # If outliers exist, handle them
    if outliers.sum() > 0:
        # Create a temporary series with outliers replaced by NaN
        temp = df['price'].copy()
        temp[outliers] = np.nan
        
        # Interpolate the NaN values
        temp = temp.interpolate(method='time')
        
        # Handle any remaining NaNs at edges
        temp = temp.fillna(method='ffill').fillna(method='bfill')
        
        # Replace original price with cleaned version
        df['price'] = temp
    
    return df

def split_train_test(df, test_size=0.2):
    """
    Splits the dataframe into training and testing sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        test_size (float): Proportion of data to use for testing
        
    Returns:
        tuple: (train_df, test_df) containing the split data
    """
    # Determine split point
    split_idx = int(len(df) * (1 - test_size))
    
    # Split the data
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df
