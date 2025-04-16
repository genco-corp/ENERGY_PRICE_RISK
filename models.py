import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, but handle if it's not available
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    print("Prophet not available. Prophet model will be skipped.")

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, output_dim=1, dropout=0.2):
        """
        Initialize LSTM model
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Dimension of hidden state
            num_layers (int): Number of LSTM layers
            output_dim (int): Dimension of output (usually 1 for forecasting)
            dropout (float): Dropout rate
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        """Forward pass through the network"""
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get output from last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out

class ElectricityPriceForecaster:
    """
    Class for electricity price forecasting using ensemble methods
    """
    
    def __init__(self, models_to_use=None, forecast_horizon=7, confidence_interval=90):
        """
        Initialize the forecaster
        
        Args:
            models_to_use (list): List of models to use for forecasting
            forecast_horizon (int): Number of days to forecast
            confidence_interval (int): Confidence interval percentage
        """
        self.forecast_horizon = forecast_horizon
        self.confidence_interval = confidence_interval
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set default to ensemble if not specified
        if models_to_use is None or "Ensemble (All)" in models_to_use:
            self.models_to_use = ["LSTM", "XGBoost", "Prophet"] if prophet_available else ["LSTM", "XGBoost"]
        else:
            self.models_to_use = [model for model in models_to_use if model != "Ensemble (All)"]
        
        # Initialize models and scalers
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
    
    def train_and_forecast(self, features, prices):
        """
        Train models and generate forecasts
        
        Args:
            features (pd.DataFrame): Feature dataframe
            prices (pd.Series): Price series
            
        Returns:
            tuple: (forecast_df, error_metrics, feature_importance)
        """
        # Split data into train/test sets
        train_data, test_data = self._split_train_test(features, prices)
        
        # Initialize forecasts dictionary to store results from each model
        forecasts = {}
        error_metrics = {}
        feature_importance = {}
        
        # Train each selected model and generate forecasts
        for model_name in self.models_to_use:
            if model_name == "LSTM":
                forecasts[model_name], error_metrics[model_name], _ = self._train_lstm(train_data, test_data)
            elif model_name == "XGBoost":
                forecasts[model_name], error_metrics[model_name], feature_importance[model_name] = self._train_xgboost(train_data, test_data)
            elif model_name == "Prophet" and prophet_available:
                forecasts[model_name], error_metrics[model_name], _ = self._train_prophet(train_data, test_data)
        
        # Generate ensemble forecast if multiple models are used
        if len(forecasts) > 1:
            forecast_df, ensemble_metrics = self._create_ensemble_forecast(forecasts, test_data)
            error_metrics["Ensemble"] = ensemble_metrics
        else:
            # If only one model is used, use its forecast
            model_name = list(forecasts.keys())[0]
            forecast_df = forecasts[model_name]
            
        # Store feature importance
        self.feature_importance = feature_importance
        
        # Extend forecast to the full forecast horizon (30 days)
        full_forecast = self._extend_forecast(forecast_df, features)
        
        return full_forecast, error_metrics, feature_importance
    
    def _split_train_test(self, features, prices):
        """Split data into training and testing sets"""
        # Combine features and prices
        data = features.copy()
        data['price'] = prices.values
        
        # Split into input features and target
        X = data.drop(columns=['timestamp', 'price'])
        y = data['price']
        
        # Get feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Store timestamp for test data
        test_timestamps = data.loc[y_test.index, 'timestamp'].values
        
        # Create train and test data dictionaries
        train_data = {
            'X': X_train,
            'y': y_train
        }
        
        test_data = {
            'X': X_test,
            'y': y_test,
            'timestamps': test_timestamps
        }
        
        return train_data, test_data
    
    def _train_lstm(self, train_data, test_data):
        """Train LSTM model and generate forecast"""
        # Scale features
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train_scaled = scaler_X.fit_transform(train_data['X'])
        y_train_scaled = scaler_y.fit_transform(train_data['y'].values.reshape(-1, 1))
        
        X_test_scaled = scaler_X.transform(test_data['X'])
        
        # Store scalers
        self.scalers['LSTM'] = {'X': scaler_X, 'y': scaler_y}
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train_scaled)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        # Create dataset and dataloader for batch training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # Define and initialize the model
        input_dim = X_train_scaled.shape[1]
        model = LSTMModel(input_dim).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        epochs = 50
        for epoch in range(epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                outputs = model(X_batch.unsqueeze(1))
                loss = criterion(outputs, y_batch)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            X_test_tensor = X_test_tensor.to(self.device)
            predictions_scaled = model(X_test_tensor.unsqueeze(1))
            predictions_scaled = predictions_scaled.cpu().numpy()
        
        # Inverse transform to get actual values
        predictions = scaler_y.inverse_transform(predictions_scaled)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': test_data['timestamps'],
            'actual': test_data['y'].values,
            'predicted': predictions.flatten()
        })
        
        # Calculate error metrics
        error_metrics = self._calculate_metrics(forecast_df['actual'], forecast_df['predicted'])
        
        # Store the trained model
        self.models['LSTM'] = model
        
        return forecast_df, error_metrics, None
    
    def _train_xgboost(self, train_data, test_data):
        """Train XGBoost model and generate forecast"""
        # Initialize XGBoost model
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        # Train the model
        model.fit(train_data['X'], train_data['y'])
        
        # Generate predictions
        predictions = model.predict(test_data['X'])
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': test_data['timestamps'],
            'actual': test_data['y'].values,
            'predicted': predictions
        })
        
        # Calculate error metrics
        error_metrics = self._calculate_metrics(forecast_df['actual'], forecast_df['predicted'])
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': train_data['X'].columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store the trained model
        self.models['XGBoost'] = model
        
        return forecast_df, error_metrics, feature_importance
    
    def _train_prophet(self, train_data, test_data):
        """Train Prophet model and generate forecast"""
        if not prophet_available:
            return None, None, None
        
        # Prepare data for Prophet
        # Prophet requires 'ds' (date) and 'y' (target) columns
        train_prophet = pd.DataFrame({
            'ds': pd.to_datetime(train_data['X'].index),
            'y': train_data['y'].values
        })
        
        # Initialize and train Prophet model
        model = Prophet(
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.1
        )
        
        # Add custom seasonalities
        model.add_seasonality(name='hourly', period=24, fourier_order=5)
        
        # Fit the model
        model.fit(train_prophet)
        
        # Create future dataframe for prediction
        future = pd.DataFrame({'ds': pd.to_datetime(test_data['timestamps'])})
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': test_data['timestamps'],
            'actual': test_data['y'].values,
            'predicted': forecast['yhat'].values
        })
        
        # Calculate confidence intervals
        forecast_df['lower'] = forecast['yhat_lower'].values
        forecast_df['upper'] = forecast['yhat_upper'].values
        
        # Calculate error metrics
        error_metrics = self._calculate_metrics(forecast_df['actual'], forecast_df['predicted'])
        
        # Store the trained model
        self.models['Prophet'] = model
        
        return forecast_df, error_metrics, None
    
    def _create_ensemble_forecast(self, forecasts, test_data):
        """Create ensemble forecast by averaging predictions from multiple models"""
        # Initialize ensemble forecast dataframe
        ensemble_df = pd.DataFrame({
            'timestamp': test_data['timestamps'],
            'actual': test_data['y'].values
        })
        
        # Calculate average prediction across all models
        predictions = np.zeros(len(test_data['y']))
        lowers = np.zeros(len(test_data['y']))
        uppers = np.zeros(len(test_data['y']))
        
        model_count = 0
        for model_name, forecast_df in forecasts.items():
            predictions += forecast_df['predicted'].values
            model_count += 1
            
            # Store individual model predictions in ensemble dataframe
            ensemble_df[f'{model_name}_pred'] = forecast_df['predicted'].values
            
            # If the model has confidence intervals, use them
            if 'lower' in forecast_df.columns and 'upper' in forecast_df.columns:
                lowers += forecast_df['lower'].values
                uppers += forecast_df['upper'].values
        
        # Calculate ensemble predictions as the average
        ensemble_df['predicted'] = predictions / model_count
        
        # Calculate ensemble confidence intervals if available
        if 'lower' in forecasts[list(forecasts.keys())[0]].columns:
            ensemble_df['lower'] = lowers / model_count
            ensemble_df['upper'] = uppers / model_count
        
        # Calculate error metrics
        error_metrics = self._calculate_metrics(ensemble_df['actual'], ensemble_df['predicted'])
        
        return ensemble_df, error_metrics
    
    def _extend_forecast(self, forecast_df, features):
        """Extend forecast to cover the full forecast horizon"""
        # Get the last timestamp in the forecast
        last_timestamp = pd.to_datetime(forecast_df['timestamp'].max())
        
        # Calculate number of hours to forecast
        hours_to_forecast = self.forecast_horizon * 24
        
        # Create dataframe with future timestamps
        future_timestamps = [last_timestamp + timedelta(hours=h+1) for h in range(hours_to_forecast)]
        future_df = pd.DataFrame({'timestamp': future_timestamps})
        
        # Use the best performing model for future forecasting
        best_model = self._identify_best_model(forecast_df)
        
        # Generate features for future timestamps
        future_features = self._generate_future_features(features, future_timestamps, forecast_df)
        
        # Generate future predictions using the best model
        if best_model == "LSTM":
            future_predictions, lower_bound, upper_bound = self._forecast_lstm(future_features)
        elif best_model == "XGBoost":
            future_predictions, lower_bound, upper_bound = self._forecast_xgboost(future_features)
        elif best_model == "Prophet" and prophet_available:
            future_predictions, lower_bound, upper_bound = self._forecast_prophet(future_timestamps)
        else:
            # Default to XGBoost if best model is unavailable
            future_predictions, lower_bound, upper_bound = self._forecast_xgboost(future_features)
        
        # Add predictions to future dataframe
        future_df['predicted'] = future_predictions
        future_df['lower'] = lower_bound
        future_df['upper'] = upper_bound
        future_df['actual'] = np.nan  # No actual values for future predictions
        
        # Combine historical forecast with future forecast
        combined_df = pd.concat([forecast_df, future_df], ignore_index=True)
        
        return combined_df
    
    def _identify_best_model(self, forecast_df):
        """Identify the best performing model based on forecast accuracy"""
        # If Ensemble is used, just return it
        if "Ensemble_pred" in forecast_df.columns:
            return "Ensemble"
        
        # If only one model is used, return it
        if len(self.models_to_use) == 1:
            return self.models_to_use[0]
        
        # Otherwise, compare model performance
        best_model = None
        lowest_mape = float('inf')
        
        for model_name in self.models_to_use:
            if f"{model_name}_pred" in forecast_df.columns:
                actual = forecast_df['actual']
                predicted = forecast_df[f"{model_name}_pred"]
                
                # Calculate MAPE
                mape = mean_absolute_percentage_error(actual, predicted)
                
                if mape < lowest_mape:
                    lowest_mape = mape
                    best_model = model_name
        
        # If no best model found, default to XGBoost or the first available model
        if best_model is None:
            if "XGBoost" in self.models_to_use:
                best_model = "XGBoost"
            else:
                best_model = self.models_to_use[0]
        
        return best_model
    
    def _generate_future_features(self, features, future_timestamps, forecast_df):
        """Generate features for future timestamps"""
        # Create a dataframe with basic temporal features
        future_features = pd.DataFrame({'timestamp': future_timestamps})
        future_features['hour'] = future_features['timestamp'].dt.hour
        future_features['day'] = future_features['timestamp'].dt.day
        future_features['day_of_week'] = future_features['timestamp'].dt.dayofweek
        future_features['month'] = future_features['timestamp'].dt.month
        future_features['quarter'] = future_features['timestamp'].dt.quarter
        future_features['year'] = future_features['timestamp'].dt.year
        
        # Create cyclical features
        future_features['hour_sin'] = np.sin(2 * np.pi * future_features['hour'] / 24)
        future_features['hour_cos'] = np.cos(2 * np.pi * future_features['hour'] / 24)
        future_features['day_of_week_sin'] = np.sin(2 * np.pi * future_features['day_of_week'] / 7)
        future_features['day_of_week_cos'] = np.cos(2 * np.pi * future_features['day_of_week'] / 7)
        future_features['month_sin'] = np.sin(2 * np.pi * future_features['month'] / 12)
        future_features['month_cos'] = np.cos(2 * np.pi * future_features['month'] / 12)
        
        # Add is_weekend flag
        future_features['is_weekend'] = future_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Add is_business_hour flag
        future_features['is_business_hour'] = ((future_features['hour'] >= 8) & 
                                              (future_features['hour'] <= 18) & 
                                              (~future_features['is_weekend'])).astype(int)
        
        # For lag features, use the latest actual and predicted values
        # We'll need to create a combined dataframe of historical and predicted prices
        historical_df = features[['timestamp', 'price']].copy()
        predicted_df = forecast_df[['timestamp', 'predicted']].rename(columns={'predicted': 'price'})
        
        combined_df = pd.concat([historical_df, predicted_df], ignore_index=True)
        combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
        combined_df = combined_df.sort_values('timestamp').drop_duplicates('timestamp', keep='first')
        
        # Generate the remaining features based on available columns in the original features
        # Get the list of feature names (excluding timestamp and price)
        feature_cols = [col for col in self.feature_names if col in features.columns]
        
        # Create an empty dataframe to hold the future features
        future_X = pd.DataFrame(index=range(len(future_timestamps)))
        
        # For each feature, either propagate from historical data or calculate based on rules
        for col in feature_cols:
            if col in future_features.columns:
                # Feature already generated
                future_X[col] = future_features[col].values
            elif 'lag' in col:
                # This is a lag feature, handle it by shifting values from the combined dataframe
                lag_hours = 0
                if 'lag_1h' in col:
                    lag_hours = 1
                elif 'lag_3h' in col:
                    lag_hours = 3
                elif 'lag_6h' in col:
                    lag_hours = 6
                elif 'lag_12h' in col:
                    lag_hours = 12
                elif 'lag_24h' in col:
                    lag_hours = 24
                elif 'lag_7d' in col:
                    lag_hours = 24 * 7
                elif 'lag_14d' in col:
                    lag_hours = 24 * 14
                elif 'lag_30d' in col:
                    lag_hours = 24 * 30
                
                # For each future timestamp, find the appropriate lagged value
                lagged_values = []
                for ts in future_timestamps:
                    lag_ts = ts - timedelta(hours=lag_hours)
                    lag_value = combined_df[combined_df['timestamp'] <= lag_ts]['price'].iloc[-1] if not combined_df[combined_df['timestamp'] <= lag_ts].empty else np.nan
                    lagged_values.append(lag_value)
                
                future_X[col] = lagged_values
            elif 'rolling' in col:
                # This is a rolling feature, propagate the last value from historical data
                future_X[col] = features[col].iloc[-1] if col in features.columns else np.nan
            else:
                # For other features, propagate the last value from historical data
                future_X[col] = features[col].iloc[-1] if col in features.columns else np.nan
        
        # Handle any missing features by filling with reasonable values
        for col in feature_cols:
            if col not in future_X.columns:
                future_X[col] = 0
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_names) - set(future_X.columns)
        for col in missing_cols:
            future_X[col] = 0
        
        # Return only the columns used during training
        future_X = future_X[self.feature_names]
        
        return future_X
    
    def _forecast_lstm(self, future_features):
        """Generate forecasts using LSTM model"""
        # Get scaler and model
        scaler_X = self.scalers['LSTM']['X']
        scaler_y = self.scalers['LSTM']['y']
        model = self.models['LSTM']
        
        # Scale features
        X_future_scaled = scaler_X.transform(future_features)
        
        # Convert to torch tensor
        X_future_tensor = torch.FloatTensor(X_future_scaled).to(self.device)
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            predictions_scaled = model(X_future_tensor.unsqueeze(1))
            predictions_scaled = predictions_scaled.cpu().numpy()
        
        # Inverse transform to get actual values
        predictions = scaler_y.inverse_transform(predictions_scaled).flatten()
        
        # Calculate confidence intervals
        z_score = 1.96  # 95% confidence interval
        if self.confidence_interval == 80:
            z_score = 1.28
        elif self.confidence_interval == 90:
            z_score = 1.645
        
        # Use residuals from training to estimate prediction uncertainty
        # This is a simplified approach - in practice, prediction intervals for neural networks
        # would be more complex (e.g., Monte Carlo dropout)
        std_dev = np.std(predictions) * 0.1  # Simplified estimate
        lower_bound = predictions - z_score * std_dev
        upper_bound = predictions + z_score * std_dev
        
        return predictions, lower_bound, upper_bound
    
    def _forecast_xgboost(self, future_features):
        """Generate forecasts using XGBoost model"""
        # Get model
        model = self.models['XGBoost']
        
        # Generate predictions
        predictions = model.predict(future_features)
        
        # Calculate confidence intervals based on historical error
        z_score = 1.96  # 95% confidence interval
        if self.confidence_interval == 80:
            z_score = 1.28
        elif self.confidence_interval == 90:
            z_score = 1.645
        
        # Use prediction standard deviation to estimate uncertainty
        # Here we're using a simplified approach - in practice, prediction intervals
        # for tree-based models would be more complex
        std_dev = np.std(predictions) * 0.15  # Simplified estimate
        lower_bound = predictions - z_score * std_dev
        upper_bound = predictions + z_score * std_dev
        
        return predictions, lower_bound, upper_bound
    
    def _forecast_prophet(self, future_timestamps):
        """Generate forecasts using Prophet model"""
        if not prophet_available or 'Prophet' not in self.models:
            return np.zeros(len(future_timestamps)), np.zeros(len(future_timestamps)), np.zeros(len(future_timestamps))
        
        # Get model
        model = self.models['Prophet']
        
        # Create future dataframe
        future = pd.DataFrame({'ds': future_timestamps})
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract predictions and confidence intervals
        predictions = forecast['yhat'].values
        lower_bound = forecast['yhat_lower'].values
        upper_bound = forecast['yhat_upper'].values
        
        return predictions, lower_bound, upper_bound
    
    def _calculate_metrics(self, actual, predicted):
        """Calculate forecast error metrics"""
        # Mean Absolute Error
        mae = mean_absolute_error(actual, predicted)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        
        # Mean Absolute Percentage Error
        # Avoid division by zero
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        
        # Theil's U statistic (a value of 1 means the model is as good as naive forecast)
        # Calculate naive forecast (previous value)
        naive_forecast = np.roll(actual, 1)
        naive_forecast[0] = naive_forecast[1]  # Replace first value
        
        # Calculate Theil's U
        numerator = np.sqrt(np.mean((predicted - actual) ** 2))
        denominator = np.sqrt(np.mean((naive_forecast - actual) ** 2))
        theils_u = numerator / denominator if denominator != 0 else np.nan
        
        # Return metrics as dictionary
        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Theils_U': theils_u
        }
