import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def engineer_features(df):
    """
    Creates features for electricity price forecasting models.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe with at least 'timestamp' and 'price' columns
        
    Returns:
        tuple: (features_df, decomposition_df) where features_df contains engineered features
               and decomposition_df contains the time series decomposition
    """
    # Make a copy to avoid modifying the original data
    data = df.copy()
    
    # Convert timestamp to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Determine the frequency by checking the most common time delta between adjacent rows
    try:
        deltas = pd.Series(data['timestamp'].diff().dropna().dt.total_seconds())
        most_common_delta = deltas.value_counts().idxmax()
        if most_common_delta == 300:  # 5 minutes = 300 seconds
            frequency = '5T'
            periods_per_hour = 12
            periods_per_day = 288  # 24 hours * 12 periods per hour
        else:
            frequency = 'H'
            periods_per_hour = 1
            periods_per_day = 24  # 24 hours per day
    except:
        # Default to hourly if we can't determine frequency
        frequency = 'H'
        periods_per_hour = 1
        periods_per_day = 24
    
    print(f"Detected data frequency: {frequency}, with {periods_per_day} periods per day")
    
    # Store the frequency metadata in a dictionary to pass to other functions
    freq_info = {
        'frequency': frequency,
        'periods_per_hour': periods_per_hour,
        'periods_per_day': periods_per_day
    }
    
    # Set timestamp as index for easier time-based operations
    data.set_index('timestamp', inplace=True)
    
    # Create basic temporal features
    features = create_temporal_features(data)
    
    # Attach frequency info to the features dataframe as a metadata attribute
    # This allows other functions to read the frequency without recalculating
    features.attrs['frequency'] = frequency
    features.attrs['periods_per_hour'] = periods_per_hour
    features.attrs['periods_per_day'] = periods_per_day
    
    # Create lagged features
    features = create_lagged_features(features)
    
    # Create rolling window features
    features = create_rolling_features(features)
    
    # Perform time series decomposition
    decomposition = decompose_time_series(data)
    
    # Create features from decomposition components if decomposition was successful
    if decomposition is not None:
        features = add_decomposition_features(features, decomposition)
    
    # Add optional external features if available
    features = add_external_features(features, data)
    
    # Reset index to have timestamp as a column again
    features.reset_index(inplace=True)
    
    # Remove rows with NaN values (due to lag features)
    features.dropna(inplace=True)
    
    return features, decomposition

def create_temporal_features(df):
    """
    Creates temporal features from timestamp index.
    
    Args:
        df (pd.DataFrame): Dataframe with timestamp index
        
    Returns:
        pd.DataFrame: Dataframe with added temporal features
    """
    features = df.copy()
    
    # Extract temporal components
    features['hour'] = features.index.hour
    features['day'] = features.index.day
    features['day_of_week'] = features.index.dayofweek
    features['month'] = features.index.month
    features['quarter'] = features.index.quarter
    features['year'] = features.index.year
    
    # Create cyclical features for hour of day
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    
    # Create cyclical features for day of week
    features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    
    # Create cyclical features for month
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # Add is_weekend flag
    features['is_weekend'] = features['day_of_week'].isin([5, 6]).astype(int)
    
    # Add is_business_hour flag (8am-6pm, Monday-Friday)
    features['is_business_hour'] = ((features['hour'] >= 8) & 
                                    (features['hour'] <= 18) & 
                                    (~features['is_weekend'])).astype(int)
    
    return features

def create_lagged_features(df):
    """
    Creates lagged features for time series forecasting.
    
    Args:
        df (pd.DataFrame): Dataframe with price column
        
    Returns:
        pd.DataFrame: Dataframe with added lagged features
    """
    features = df.copy()
    
    # Determine frequency of data to set appropriate lags
    # Try to infer from the index
    try:
        # Get the most common time difference between consecutive timestamps
        deltas = pd.Series(pd.to_datetime(features.index).to_series().diff().dropna().dt.total_seconds())
        most_common_delta = deltas.value_counts().idxmax()
        
        # Calculate periods per hour and day for the detected frequency
        if most_common_delta == 300:  # 5 minutes
            periods_per_hour = 12
            periods_per_day = 288  # 24 hours * 12 periods per hour
            frequency = '5T'
        else:  # Default to hourly
            periods_per_hour = 1
            periods_per_day = 24
            frequency = 'H'
    except:
        # Default to hourly if inference fails
        periods_per_hour = 1
        periods_per_day = 24
        frequency = 'H'
    
    print(f"Creating lagged features with detected frequency: {frequency} ({periods_per_day} periods per day)")
    
    # Create lagged features scaled by frequency
    # Convert concepts of hours to periods
    for hours in [1, 3, 6, 12, 24]:
        lag_periods = hours * periods_per_hour
        features[f'price_lag_{hours}h'] = features['price'].shift(lag_periods)
    
    # Create day-of-week lags (same time last week, two weeks ago)
    for days in [7, 14]:
        lag_periods = days * periods_per_day
        features[f'price_lag_{days}d'] = features['price'].shift(lag_periods)
    
    # Create month lag (same time one month ago - approximate)
    features['price_lag_30d'] = features['price'].shift(30 * periods_per_day)
    
    return features

def create_rolling_features(df):
    """
    Creates rolling window statistical features.
    
    Args:
        df (pd.DataFrame): Dataframe with price column
        
    Returns:
        pd.DataFrame: Dataframe with added rolling features
    """
    features = df.copy()
    
    # Determine frequency of data to set appropriate window sizes
    # Try to infer from the index
    try:
        # Get the most common time difference between consecutive timestamps
        deltas = pd.Series(pd.to_datetime(features.index).to_series().diff().dropna().dt.total_seconds())
        most_common_delta = deltas.value_counts().idxmax()
        
        # Calculate periods per day for the detected frequency
        if most_common_delta == 300:  # 5 minutes
            periods_per_hour = 12
            periods_per_day = 288  # 24 hours * 12 periods per hour
            frequency = '5T'
        else:  # Default to hourly
            periods_per_hour = 1
            periods_per_day = 24
            frequency = 'H'
    except:
        # Default to hourly if inference fails
        periods_per_hour = 1
        periods_per_day = 24
        frequency = 'H'
    
    print(f"Creating rolling features with detected frequency: {frequency} ({periods_per_day} periods per day)")
    
    # Define window sizes in terms of periods
    windows = [
        (1, periods_per_day),  # 1 day
        (7, periods_per_day * 7),  # 1 week
        (30, periods_per_day * 30)  # 1 month (approximate)
    ]
    
    # Create rolling window statistics scaled by frequency
    for day_count, window_size in windows:
        window_name = f"{day_count}d"
        
        # Mean
        features[f'price_rolling_mean_{window_name}'] = features['price'].rolling(
            window=window_size, min_periods=1).mean()
        
        # Standard deviation
        features[f'price_rolling_std_{window_name}'] = features['price'].rolling(
            window=window_size, min_periods=1).std()
        
        # Min/Max
        features[f'price_rolling_min_{window_name}'] = features['price'].rolling(
            window=window_size, min_periods=1).min()
        features[f'price_rolling_max_{window_name}'] = features['price'].rolling(
            window=window_size, min_periods=1).max()
        
        # Range (max - min)
        features[f'price_rolling_range_{window_name}'] = (
            features[f'price_rolling_max_{window_name}'] - 
            features[f'price_rolling_min_{window_name}']
        )
    
    return features

def decompose_time_series(df):
    """
    Performs time series decomposition on the price data.
    
    Args:
        df (pd.DataFrame): Dataframe with price column and timestamp index
        
    Returns:
        pd.DataFrame: Dataframe with decomposition components
    """
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
        
        # Determine frequency of data to set appropriate decomposition period
        try:
            # Get the most common time difference between consecutive timestamps
            deltas = pd.Series(pd.to_datetime(df.index).to_series().diff().dropna().dt.total_seconds())
            most_common_delta = deltas.value_counts().idxmax()
            
            # Calculate periods per day for the detected frequency
            if most_common_delta == 300:  # 5 minutes
                periods_per_day = 288  # 24 hours * 12 periods per hour
                # Weekly seasonality for 5-minute data (288 periods per day * 7 days)
                seasonality_period = periods_per_day * 7
            else:  # Default to hourly
                periods_per_day = 24
                # Weekly seasonality for hourly data (24 periods per day * 7 days)
                seasonality_period = periods_per_day * 7
        except:
            # Default to hourly if inference fails
            periods_per_day = 24
            seasonality_period = periods_per_day * 7
        
        print(f"Using time series decomposition with seasonality period: {seasonality_period}")
        
        # Check if we have enough data for decomposition (at least 2 full periods)
        if len(df) < 2 * seasonality_period:
            print(f"Not enough data for decomposition: {len(df)} points, need at least {2 * seasonality_period}")
            return None
        
        # Perform decomposition
        result = seasonal_decompose(
            df['price'], 
            model='additive', 
            period=seasonality_period,
            extrapolate_trend='freq'
        )
        
        # Combine decomposition results
        decomposition = pd.DataFrame({
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid,
            'price': df['price']
        })
        
        return decomposition
    
    except Exception as e:
        print(f"Error in time series decomposition: {str(e)}")
        return None

def add_decomposition_features(features_df, decomposition_df):
    """
    Adds decomposition components as features.
    
    Args:
        features_df (pd.DataFrame): Dataframe with features
        decomposition_df (pd.DataFrame): Dataframe with decomposition components
        
    Returns:
        pd.DataFrame: Combined dataframe with decomposition features
    """
    if decomposition_df is None:
        return features_df
    
    # Combine the dataframes on index
    features = features_df.copy()
    
    # Add decomposition components as features
    features['trend'] = decomposition_df['trend']
    features['seasonal'] = decomposition_df['seasonal']
    features['residual'] = decomposition_df['residual']
    
    # Calculate additional metrics from decomposition
    features['trend_pct'] = (decomposition_df['trend'] / decomposition_df['price']).replace([np.inf, -np.inf], np.nan).fillna(0)
    features['seasonal_pct'] = (decomposition_df['seasonal'] / decomposition_df['price']).replace([np.inf, -np.inf], np.nan).fillna(0)
    features['residual_pct'] = (decomposition_df['residual'] / decomposition_df['price']).replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return features

def add_external_features(features_df, data_df):
    """
    Adds external features if available in the original data.
    
    Args:
        features_df (pd.DataFrame): Dataframe with features
        data_df (pd.DataFrame): Original dataframe with potential external features
        
    Returns:
        pd.DataFrame: Dataframe with added external features
    """
    features = features_df.copy()
    
    # Determine frequency of data to set appropriate window sizes
    try:
        # Get the most common time difference between consecutive timestamps
        deltas = pd.Series(pd.to_datetime(data_df.index).to_series().diff().dropna().dt.total_seconds())
        most_common_delta = deltas.value_counts().idxmax()
        
        # Calculate periods per day for the detected frequency
        if most_common_delta == 300:  # 5 minutes
            periods_per_hour = 12
            periods_per_day = 288  # 24 hours * 12 periods per hour
        else:  # Default to hourly
            periods_per_hour = 1
            periods_per_day = 24
    except:
        # Default to hourly if inference fails
        periods_per_hour = 1
        periods_per_day = 24
    
    print(f"Creating external features with {periods_per_day} periods per day")
    
    # List of potential external features to check
    external_features = [
        'temperature', 'wind_speed', 'solar_irradiance', 
        'precipitation', 'load'
    ]
    
    # Add external features if available
    for feature in external_features:
        if feature in data_df.columns:
            features[feature] = data_df[feature]
            
            # Create rolling statistics for each external feature (1 day window)
            features[f'{feature}_rolling_mean_1d'] = data_df[feature].rolling(
                window=periods_per_day, min_periods=1).mean()
            features[f'{feature}_rolling_std_1d'] = data_df[feature].rolling(
                window=periods_per_day, min_periods=1).std()
            
            # Create lag features for external variables (24 hour lag)
            features[f'{feature}_lag_24h'] = data_df[feature].shift(periods_per_day)
    
    return features
