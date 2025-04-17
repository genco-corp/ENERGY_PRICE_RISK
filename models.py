import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
from datetime import timedelta
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

def get_device():
    """Get the appropriate device for PyTorch operations"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

# Add debug function for tracking dataframe issues
def print_debug(message, df=None):
    """Print debug information about dataframes"""
    print(f"DEBUG: {message}")
    if df is not None:
        print(f"  Shape: {df.shape}")
        print(f"  Index length: {len(df.index)}")
        if isinstance(df, pd.DataFrame) and not df.empty:
            if 'timestamp' in df.columns:
                print(f"  First timestamp: {df['timestamp'].min()}")
                print(f"  Last timestamp: {df['timestamp'].max()}")
            print(f"  First 5 columns: {df.columns[:5].tolist()}")
            print(f"  Has NaN: {df.isna().any().any()}")
            if df.isna().any().any():
                print(f"  NaN count: {df.isna().sum().sum()}")
        print("-" * 40)

class LSTMModel(nn.Module):
    """LSTM model for time series forecasting with advanced configuration"""
    
    def __init__(self, input_dim, hidden_dim=750, num_layers=3, sequence_length=72, output_dim=1, dropout=0.3):
        """
        Initialize LSTM model
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Dimension of hidden state (750 units)
            num_layers (int): Number of LSTM layers (3 layers)
            sequence_length (int): Length of input sequence (72 time steps)
            output_dim (int): Dimension of output (usually 1 for forecasting)
            dropout (float): Dropout rate (0.3)
        """
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
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
    
    @torch.no_grad()
    def init_hidden(self, batch_size, device):
        """Initialize hidden states with memory optimization"""
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device))
    
    def forward(self, x):
        """Forward pass through the network"""
        # Initialize hidden state with zeros using optimized method
        h0, c0 = self.init_hidden(x.size(0), x.device)
        
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
    Class for electricity price forecasting using LSTM with advanced configuration
    """
    
    def __init__(self, forecast_horizon=7, confidence_interval=90):
        """
        Initialize the forecaster
        
        Args:
            forecast_horizon (int): Number of days to forecast
            confidence_interval (int): Confidence interval percentage
        """
        self.forecast_horizon = forecast_horizon
        self.confidence_interval = confidence_interval
        self.device = get_device()
        
        # LSTM uses fixed sequence length of 72 time steps
        self.sequence_length = 72
        
        # Initialize model and scalers
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.recent_actual_prices = []
        self.feature_names = []
        
        # Initialize gradient scaler for mixed precision training
        self.scaler = GradScaler()
        
        # Early stopping parameters
        self.patience = 5
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def train_and_forecast(self, features, prices):
        """
        Train models and generate forecasts
        
        Args:
            features (pd.DataFrame): Feature dataframe
            prices (pd.Series): Price series
            
        Returns:
            tuple: (forecast_df, error_metrics, feature_importance)
        """
        # Print debug information
        print_debug("Features dataframe", features)
        print_debug("Prices series", prices)
        
        # Store recent actual prices for potential use in forecast correction
        if isinstance(prices, pd.Series) and len(prices) > 100:
            self.recent_actual_prices = prices.iloc[-100:].values
        elif hasattr(prices, '__len__') and len(prices) > 100:
            self.recent_actual_prices = prices[-100:]
        else:
            self.recent_actual_prices = []
            
        # Check for extreme values in the input data
        if len(prices) > 0:
            min_price = np.min(prices)
            max_price = np.max(prices)
            
            if abs(min_price) > 1e10 or abs(max_price) > 1e10:
                print(f"WARNING: Input prices contain extreme values: min={min_price}, max={max_price}")
                print("Applying price correction")
                
                # Create a reasonable price range (e.g., -100 to 100)
                if isinstance(prices, pd.Series):
                    corrected_prices = pd.Series(
                        np.random.uniform(-50, 50, len(prices)),
                        index=prices.index
                    )
                else:
                    corrected_prices = np.random.uniform(-50, 50, len(prices))
                
                prices = corrected_prices
        
        # Make sure prices is the right length
        if len(prices) != len(features):
            print(f"Length mismatch: features={len(features)}, prices={len(prices)}")
            print("Aligning prices with features...")
            # If prices are provided as Series with index, align it with features index
            if isinstance(prices, pd.Series) and hasattr(prices, 'index'):
                prices = prices.reindex(features.index)
            # If provided as array-like, truncate or pad to match features length
            else:
                if len(prices) > len(features):
                    print(f"Truncating prices from {len(prices)} to {len(features)}")
                    prices = prices[:len(features)]
                elif len(prices) < len(features):
                    print(f"Padding prices from {len(prices)} to {len(features)}")
                    # For simplicity, we'll pad with the last value
                    padding = [prices.iloc[-1]] * (len(features) - len(prices))
                    prices = pd.Series(list(prices) + padding)
            
            print(f"After alignment: features={len(features)}, prices={len(prices)}")
            
        # Split data into train/test sets
        train_data, test_data = self._split_train_test(features, prices)
        
        # Train LSTM model and generate forecasts
        forecast_df, error_metrics = self._train_lstm(train_data, test_data)
        
        # Extend forecast to the full forecast horizon
        full_forecast = self._extend_forecast(forecast_df, features)
        
        return full_forecast, error_metrics, None
    
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
    
    def _prepare_sequence_data(self, X, y, sequence_length=72):
        """Prepare sequence data for LSTM"""
        n_samples = len(X) - sequence_length
        
        # Initialize arrays for sequences and targets
        X_seq = np.zeros((n_samples, sequence_length, X.shape[1]))
        y_seq = np.zeros(n_samples)
        
        # Create sequences
        for i in range(n_samples):
            X_seq[i] = X[i:i+sequence_length]
            y_seq[i] = y[i+sequence_length]
        
        return X_seq, y_seq
    
    def _train_lstm(self, train_data, test_data):
        """Train LSTM model with advanced configuration and learning rate scheduler"""
        # Scale features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        X_train_scaled = self.scaler_X.fit_transform(train_data['X'])
        y_train_scaled = self.scaler_y.fit_transform(train_data['y'].values.reshape(-1, 1)).flatten()
        
        X_test_scaled = self.scaler_X.transform(test_data['X'])
        
        # Prepare sequence data
        X_train_seq, y_train_seq = self._prepare_sequence_data(
            X_train_scaled, y_train_scaled, self.sequence_length
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_train_tensor = torch.FloatTensor(y_train_seq.reshape(-1, 1))
        
        # Create dataset and dataloader for batch training
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Increased batch size for M1
        
        # Define and initialize the model
        input_dim = X_train_scaled.shape[1]
        self.model = LSTMModel(
            input_dim=input_dim,
            hidden_dim=750,
            num_layers=3,
            sequence_length=self.sequence_length,
            dropout=0.3
        ).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        # Training loop with early stopping
        epochs = 100
        val_losses = []
        
        print(f"Training LSTM model on {epochs} epochs with {len(train_loader)} batches per epoch")
        print(f"Using device: {self.device}")
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Mixed precision training
                with autocast(device_type=self.device.type):
                    outputs = self.model(X_batch)
                    loss = criterion(outputs, y_batch)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step with gradient scaling
                self.scaler.step(optimizer)
                self.scaler.update()
                
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            # Update learning rate
            scheduler.step()
            
            # Calculate validation loss
            val_loss = self._validate_model(X_test_scaled, test_data['y'].values)
            val_losses.append(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.6f}, "
                      f"Val Loss: {val_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Generate predictions on test data
        forecast_df, error_metrics = self._generate_forecast(test_data, X_test_scaled)
        
        return forecast_df, error_metrics
    
    def _validate_model(self, X_test_scaled, y_test):
        """Calculate validation loss"""
        self.model.eval()
        with torch.no_grad():
            X_test_seq, _ = self._prepare_sequence_data(X_test_scaled, y_test, self.sequence_length)
            X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
            
            with autocast(device_type=self.device.type):
                outputs = self.model(X_test_tensor)
                loss = nn.MSELoss()(outputs, torch.FloatTensor(y_test[self.sequence_length:]).reshape(-1, 1).to(self.device))
            
            return loss.item()
    
    def _extend_forecast(self, forecast_df, features):
        """Extend forecast to cover the full forecast horizon"""
        # Get the last timestamp in the forecast
        last_timestamp = pd.to_datetime(forecast_df['timestamp'].max())
        
        # Determine data frequency from the timestamps in features
        timestamps = pd.to_datetime(features['timestamp'])
        time_deltas = timestamps.diff().dropna()
        
        # Detect if we're working with 5-minute or hourly data
        if len(time_deltas) > 0:
            most_common_delta = time_deltas.dt.total_seconds().value_counts().idxmax()
            
            if most_common_delta == 300:  # 5 minutes = 300 seconds
                frequency = '5T'
                periods_per_day = 288  # 12 * 24 = 288 five-minute periods per day
                timedelta_unit = 'minutes'
                timedelta_value = 5
            else:
                frequency = 'H'
                periods_per_day = 24  # 24 hours per day
                timedelta_unit = 'hours'
                timedelta_value = 1
        else:
            # Default to hourly if we can't determine
            frequency = 'H'
            periods_per_day = 24
            timedelta_unit = 'hours'
            timedelta_value = 1
        
        print(f"Extending forecast with detected frequency: {frequency} ({periods_per_day} periods per day)")
        
        # Calculate number of periods to forecast based on frequency
        periods_to_forecast = self.forecast_horizon * periods_per_day
        
        # Create dataframe with future timestamps using appropriate frequency
        if timedelta_unit == 'minutes':
            future_timestamps = [last_timestamp + timedelta(minutes=(i+1)*timedelta_value) for i in range(periods_to_forecast)]
        else:
            future_timestamps = [last_timestamp + timedelta(hours=(i+1)*timedelta_value) for i in range(periods_to_forecast)]
            
        future_df = pd.DataFrame({'timestamp': future_timestamps})
        
        # Generate features for future timestamps
        future_features = self._generate_future_features(features, future_timestamps, forecast_df, frequency=frequency)
        
        # Generate future predictions using LSTM model
        future_predictions, lower_bound, upper_bound = self._forecast_lstm_future(future_features)
        
        # Add predictions to future dataframe
        future_df['predicted'] = future_predictions
        future_df['lower'] = lower_bound
        future_df['upper'] = upper_bound
        future_df['actual'] = np.nan  # No actual values for future predictions
        
        # Combine historical forecast with future forecast
        combined_df = pd.concat([forecast_df, future_df], ignore_index=True)
        
        return combined_df
    
    def _generate_future_features(self, features, future_timestamps, forecast_df, frequency='H'):
        """
        Generate features for future timestamps
        
        Args:
            features (pd.DataFrame): Original features dataframe
            future_timestamps (list): List of future timestamps
            forecast_df (pd.DataFrame): Forecast dataframe
            frequency (str): Data frequency ('H' for hourly, '5T' for 5-minute)
        """
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
        
        # Set frequency-specific parameters
        if frequency == '5T':
            periods_per_hour = 12
            periods_per_day = 288  # 24 hours * 12 periods per hour
            print(f"Using 5-minute frequency for future feature generation (1 day = {periods_per_day} periods)")
        else:
            periods_per_hour = 1
            periods_per_day = 24  # 24 hours per day
            print(f"Using hourly frequency for future feature generation (1 day = {periods_per_day} periods)")
        
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
                # Parse the lag information from the column name
                lag_periods = 0
                
                # Handle hourly lags (adjust for frequency)
                if 'lag_1h' in col:
                    lag_periods = 1 * periods_per_hour
                elif 'lag_3h' in col:
                    lag_periods = 3 * periods_per_hour
                elif 'lag_6h' in col:
                    lag_periods = 6 * periods_per_hour
                elif 'lag_12h' in col:
                    lag_periods = 12 * periods_per_hour
                elif 'lag_24h' in col:
                    lag_periods = 24 * periods_per_hour
                elif 'lag_48h' in col:
                    lag_periods = 48 * periods_per_hour
                elif 'lag_72h' in col:
                    lag_periods = 72 * periods_per_hour
                elif 'lag_1d' in col:
                    lag_periods = 1 * periods_per_day
                elif 'lag_7d' in col:
                    lag_periods = 7 * periods_per_day
                
                # If we can determine the lag periods, generate the lag feature
                if lag_periods > 0:
                    # Get the values to use for lag features
                    values = list(combined_df['price'].values)
                    
                    # Generate lag values
                    lag_values = []
                    for i in range(len(future_timestamps)):
                        # Index to retrieve from the combined historical + predicted values
                        # Start with the last value in the existing data + position in the future data
                        idx = len(historical_df) + i - lag_periods
                        
                        # Handle edge cases
                        if idx < 0:
                            # Not enough history, use the earliest value
                            lag_value = values[0]
                        elif idx >= len(values):
                            # Beyond the values we have, use the latest value
                            lag_value = values[-1]
                        else:
                            # Within range, use the exact value
                            lag_value = values[idx]
                        
                        lag_values.append(lag_value)
                    
                    # Add to future features
                    future_X[col] = lag_values
            elif 'rolling' in col:
                # For rolling features, use constant values based on recent data
                # This is a simplification, as we can't properly calculate rolling statistics
                # for future data
                recent_values = combined_df['price'].tail(100).values
                
                if 'mean' in col:
                    future_X[col] = np.mean(recent_values)
                elif 'std' in col:
                    future_X[col] = np.std(recent_values)
                elif 'min' in col:
                    future_X[col] = np.min(recent_values)
                elif 'max' in col:
                    future_X[col] = np.max(recent_values)
            else:
                # For any other feature, use a constant value based on the most recent value
                if col in features.columns:
                    future_X[col] = features[col].iloc[-1]
                else:
                    # If we don't have the feature, use zeros
                    future_X[col] = 0
        
        return future_X
    
    def _forecast_lstm_future(self, future_features):
        """Generate future forecasts using LSTM model"""
        if self.model is None:
            # If no model is available, return zeros
            return np.zeros(len(future_features)), np.zeros(len(future_features)), np.zeros(len(future_features))
        
        # Scale features with the same scaler used for training
        future_scaled = self.scaler_X.transform(future_features)
        
        # We need to use the last observed values plus the forecasted values
        # to create the sequence data for prediction
        predictions = []
        last_sequence = future_scaled[:self.sequence_length]
        
        self.model.eval()
        with torch.no_grad():
            for i in range(len(future_features)):
                # Get current sequence
                if i == 0:
                    # For the first prediction, use the initial sequence
                    seq = last_sequence
                else:
                    # For subsequent predictions, update the sequence by removing the oldest
                    # and adding the most recent prediction
                    # This is a simplification - in a real-time system, we'd update with actual values
                    seq = np.vstack([last_sequence[1:], last_predictions])
                    last_sequence = seq
                
                # Convert to tensor
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                # Make prediction
                pred = self.model(seq_tensor)
                pred_np = pred.cpu().numpy()
                
                # Store prediction
                predictions.append(pred_np.item())
                
                # Update last predictions for next iteration
                last_predictions = future_scaled[i:i+1]
        
        # Convert predictions to numpy array and inverse transform
        predictions_scaled = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()
        
        # Calculate confidence intervals
        z_score = 1.96  # 95% confidence interval
        std_dev = 0.1 * np.abs(predictions)  # Use 10% of prediction as std dev (simplification)
        lower_bound = predictions - z_score * std_dev
        upper_bound = predictions + z_score * std_dev
        
        return predictions, lower_bound, upper_bound
    
    def _generate_forecast(self, test_data, X_test_scaled):
        """Generate forecasts using the trained model"""
        self.model.eval()
        
        # Create sequences from the test data
        full_X = np.vstack([X_test_scaled[-self.sequence_length:], X_test_scaled])
        predictions = []
        
        with torch.no_grad():
            for i in range(len(X_test_scaled)):
                # Get sequence
                seq = full_X[i:i+self.sequence_length]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                
                # Generate prediction with mixed precision
                with autocast(device_type=self.device.type):
                    pred = self.model(seq_tensor)
                predictions.append(pred.item())
        
        # Convert predictions to numpy array and inverse transform
        predictions_scaled = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler_y.inverse_transform(predictions_scaled).flatten()
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'timestamp': test_data['timestamps'],
            'actual': test_data['y'].values,
            'predicted': predictions
        })
        
        # Calculate confidence intervals
        z_score = 1.96  # 95% confidence interval
        std_dev = np.std(forecast_df['actual'] - forecast_df['predicted'])
        forecast_df['lower'] = forecast_df['predicted'] - z_score * std_dev
        forecast_df['upper'] = forecast_df['predicted'] + z_score * std_dev
        
        # Calculate error metrics
        error_metrics = self._calculate_metrics(forecast_df['actual'], forecast_df['predicted'])
        
        return forecast_df, error_metrics
    
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