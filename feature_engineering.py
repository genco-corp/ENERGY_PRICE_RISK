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
    
    # Set timestamp as index for easier time-based operations
    data.set_index('timestamp', inplace=True)
    
    # Create basic temporal features
    features = create_temporal_features(data)
    
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
    
    # Create lagged features (previous hours)
    for lag in [1, 3, 6, 12, 24]:
        features[f'price_lag_{lag}h'] = features['price'].shift(lag)
    
    # Create day-of-week lags (same hour last week, two weeks ago)
    for lag in [24*7, 24*14]:
        features[f'price_lag_{lag//24}d'] = features['price'].shift(lag)
    
    # Create month lag (same hour one month ago - approximate)
    features['price_lag_30d'] = features['price'].shift(24*30)
    
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
    
    # Create rolling window statistics
    for window in [24, 24*7, 24*30]:  # 1 day, 1 week, 1 month
        window_name = f"{window//24}d" if window >= 24 else f"{window}h"
        
        # Mean
        features[f'price_rolling_mean_{window_name}'] = features['price'].rolling(
            window=window, min_periods=1).mean()
        
        # Standard deviation
        features[f'price_rolling_std_{window_name}'] = features['price'].rolling(
            window=window, min_periods=1).std()
        
        # Min/Max
        features[f'price_rolling_min_{window_name}'] = features['price'].rolling(
            window=window, min_periods=1).min()
        features[f'price_rolling_max_{window_name}'] = features['price'].rolling(
            window=window, min_periods=1).max()
        
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
        
        # Check if we have enough data for decomposition (at least 2 full periods)
        if len(df) < 2 * 24 * 7:  # 2 weeks of hourly data
            return None
        
        # Perform decomposition
        result = seasonal_decompose(
            df['price'], 
            model='additive', 
            period=24*7,  # Weekly seasonality for hourly data
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
    
    # List of potential external features to check
    external_features = [
        'temperature', 'wind_speed', 'solar_irradiance', 
        'precipitation', 'load'
    ]
    
    # Add external features if available
    for feature in external_features:
        if feature in data_df.columns:
            features[feature] = data_df[feature]
            
            # Create rolling statistics for each external feature
            features[f'{feature}_rolling_mean_1d'] = data_df[feature].rolling(window=24, min_periods=1).mean()
            features[f'{feature}_rolling_std_1d'] = data_df[feature].rolling(window=24, min_periods=1).std()
            
            # Create lag features for external variables
            features[f'{feature}_lag_24h'] = data_df[feature].shift(24)
    
    return features
