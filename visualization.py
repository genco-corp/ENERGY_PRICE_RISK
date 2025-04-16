import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

def plot_historical_prices(data):
    """
    Plot historical electricity prices
    
    Args:
        data (pd.DataFrame): DataFrame with timestamp and price columns
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Create figure
    fig = go.Figure()
    
    # Add trace for price
    fig.add_trace(go.Scatter(
        x=data['timestamp'],
        y=data['price'],
        mode='lines',
        name='Historical Price',
        line=dict(color='royalblue', width=1.5)
    ))
    
    # Update layout
    fig.update_layout(
        title='Historical Electricity Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def plot_price_forecast(forecast_df, historical_df, confidence_interval=90):
    """
    Plot price forecast with confidence intervals
    
    Args:
        forecast_df (pd.DataFrame): DataFrame with forecast results
        historical_df (pd.DataFrame): DataFrame with historical data
        confidence_interval (int): Confidence interval percentage for visualization
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Convert timestamp to datetime if needed
    if not pd.api.types.is_datetime64_any_dtype(forecast_df['timestamp']):
        forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
    
    # Get the date where actual values end and only predictions begin
    forecast_start = forecast_df[forecast_df['actual'].isna()]['timestamp'].min()
    
    # Split into historical forecast (with actual values) and future forecast
    if forecast_start is not None:
        historical_forecast = forecast_df[forecast_df['timestamp'] < forecast_start].copy()
        future_forecast = forecast_df[forecast_df['timestamp'] >= forecast_start].copy()
    else:
        historical_forecast = forecast_df.copy()
        future_forecast = pd.DataFrame(columns=forecast_df.columns)
    
    # Create figure
    fig = go.Figure()
    
    # Add historical actual prices (blue line)
    fig.add_trace(go.Scatter(
        x=historical_forecast['timestamp'],
        y=historical_forecast['actual'],
        mode='lines',
        name='Actual Price',
        line=dict(color='royalblue', width=2)
    ))
    
    # Add historical forecast (red line)
    fig.add_trace(go.Scatter(
        x=historical_forecast['timestamp'],
        y=historical_forecast['predicted'],
        mode='lines',
        name='Historical Forecast',
        line=dict(color='crimson', width=2)
    ))
    
    # Add future forecast (green line)
    if not future_forecast.empty:
        fig.add_trace(go.Scatter(
            x=future_forecast['timestamp'],
            y=future_forecast['predicted'],
            mode='lines',
            name='Future Forecast',
            line=dict(color='forestgreen', width=2.5)
        ))
    
    # Add confidence intervals as a shaded area
    if 'lower' in forecast_df.columns and 'upper' in forecast_df.columns:
        # Add confidence intervals for historical forecast
        fig.add_trace(go.Scatter(
            x=historical_forecast['timestamp'],
            y=historical_forecast['upper'],
            mode='lines',
            name=f'{confidence_interval}% Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=historical_forecast['timestamp'],
            y=historical_forecast['lower'],
            mode='lines',
            name=f'{confidence_interval}% Lower Bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(220, 20, 60, 0.2)',
            showlegend=False
        ))
        
        # Add confidence intervals for future forecast
        if not future_forecast.empty:
            fig.add_trace(go.Scatter(
                x=future_forecast['timestamp'],
                y=future_forecast['upper'],
                mode='lines',
                name=f'{confidence_interval}% Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=future_forecast['timestamp'],
                y=future_forecast['lower'],
                mode='lines',
                name=f'{confidence_interval}% Lower Bound',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(34, 139, 34, 0.2)',
                showlegend=True
            ))
    
    # Update layout
    fig.update_layout(
        title=f'Electricity Price Forecast with {confidence_interval}% Confidence Interval',
        xaxis_title='Date',
        yaxis_title='Price',
        template='plotly_white',
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=7, label="1w", step="day", stepmode="backward"),
                dict(count=14, label="2w", step="day", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def plot_error_metrics(error_metrics):
    """
    Plot error metrics for each model
    
    Args:
        error_metrics (dict): Dictionary of error metrics by model
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Create lists to store data for plotting
    models = []
    mae_values = []
    rmse_values = []
    mape_values = []
    theils_u_values = []
    
    # Extract values from the dictionary
    for model, metrics in error_metrics.items():
        models.append(model)
        mae_values.append(metrics['MAE'])
        rmse_values.append(metrics['RMSE'])
        mape_values.append(metrics['MAPE'])
        theils_u_values.append(metrics['Theils_U'])
    
    # Create subplots with 4 metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Mean Absolute Error (MAE)', 
                        'Root Mean Square Error (RMSE)',
                        'Mean Absolute Percentage Error (MAPE)',
                        "Theil's U Statistic")
    )
    
    # Add bars for each metric
    fig.add_trace(
        go.Bar(x=models, y=mae_values, name='MAE', marker_color='royalblue'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='crimson'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=models, y=mape_values, name='MAPE (%)', marker_color='forestgreen'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=theils_u_values, name="Theil's U", marker_color='darkorange'),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        title='Model Performance Metrics',
        template='plotly_white',
        showlegend=False,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Format y-axes
    fig.update_yaxes(title_text='MAE', row=1, col=1)
    fig.update_yaxes(title_text='RMSE', row=1, col=2)
    fig.update_yaxes(title_text='MAPE (%)', row=2, col=1)
    fig.update_yaxes(title_text="Theil's U", row=2, col=2)
    
    return fig

def plot_feature_importance(feature_importance):
    """
    Plot feature importance from models
    
    Args:
        feature_importance (dict or pd.DataFrame): Feature importance data
        
    Returns:
        go.Figure: Plotly figure object
    """
    # If feature_importance is a dictionary, use the first model's importance
    if isinstance(feature_importance, dict):
        if not feature_importance:
            # Empty dictionary
            return go.Figure().update_layout(
                title='No Feature Importance Data Available',
                template='plotly_white'
            )
        
        # Get the model with feature importance data
        model_name = list(feature_importance.keys())[0]
        importance_df = feature_importance[model_name]
    else:
        # Already a DataFrame
        importance_df = feature_importance
    
    # Sort by importance and take top 15 features
    importance_df = importance_df.sort_values('importance', ascending=False).head(15)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=importance_df['feature'],
        x=importance_df['importance'],
        orientation='h',
        marker_color='royalblue'
    ))
    
    # Update layout
    fig.update_layout(
        title='Top 15 Features by Importance',
        xaxis_title='Importance',
        yaxis_title='Feature',
        template='plotly_white',
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        yaxis=dict(autorange="reversed")  # Reverse y-axis to show highest importance at top
    )
    
    return fig

def plot_decomposition(decomposition_df):
    """
    Plot time series decomposition components
    
    Args:
        decomposition_df (pd.DataFrame): Dataframe with decomposition components
        
    Returns:
        go.Figure: Plotly figure object
    """
    if decomposition_df is None:
        # Return empty figure if no decomposition data
        return go.Figure().update_layout(
            title='No Decomposition Data Available',
            template='plotly_white'
        )
    
    # Create subplots - one for each component
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Original', 'Trend', 'Seasonal', 'Residual'),
        shared_xaxes=True,
        vertical_spacing=0.05
    )
    
    # Add traces for each component
    fig.add_trace(
        go.Scatter(x=decomposition_df.index, y=decomposition_df['price'], 
                  mode='lines', name='Original', line=dict(color='royalblue')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition_df.index, y=decomposition_df['trend'], 
                  mode='lines', name='Trend', line=dict(color='crimson')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition_df.index, y=decomposition_df['seasonal'], 
                  mode='lines', name='Seasonal', line=dict(color='forestgreen')),
        row=3, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=decomposition_df.index, y=decomposition_df['residual'], 
                  mode='lines', name='Residual', line=dict(color='darkorange')),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title='Time Series Decomposition of Electricity Prices',
        template='plotly_white',
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode='x unified'
    )
    
    # Add range slider to bottom subplot only
    fig.update_xaxes(rangeslider_visible=True, row=4, col=1)
    
    # Y-axis titles
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Trend', row=2, col=1)
    fig.update_yaxes(title_text='Seasonal', row=3, col=1)
    fig.update_yaxes(title_text='Residual', row=4, col=1)
    
    return fig
