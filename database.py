import os
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import datetime
import json

# Create SQLAlchemy base
Base = declarative_base()

# Models
class Dataset(Base):
    """Table for storing uploaded datasets"""
    __tablename__ = 'datasets'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    start_date = Column(DateTime, nullable=True)
    end_date = Column(DateTime, nullable=True)
    record_count = Column(Integer, nullable=False)
    has_external_features = Column(Boolean, default=False)
    
    # Relationships
    price_data = relationship("PriceData", back_populates="dataset", cascade="all, delete-orphan")
    forecasts = relationship("Forecast", back_populates="dataset", cascade="all, delete-orphan")

class PriceData(Base):
    """Table for storing electricity price data"""
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    price = Column(Float, nullable=False)
    temperature = Column(Float, nullable=True)
    wind_speed = Column(Float, nullable=True)
    solar_irradiance = Column(Float, nullable=True)
    precipitation = Column(Float, nullable=True)
    load = Column(Float, nullable=True)
    
    # Relationships
    dataset = relationship("Dataset", back_populates="price_data")

class Forecast(Base):
    """Table for storing forecast results"""
    __tablename__ = 'forecasts'
    
    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey('datasets.id'), nullable=False)
    name = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    forecast_horizon = Column(Integer, nullable=False)  # in days
    confidence_interval = Column(Integer, nullable=False)  # e.g. 90 for 90%
    models_used = Column(String(255), nullable=False)  # comma-separated list
    
    # Serialized data
    error_metrics = Column(Text, nullable=True)  # JSON string
    feature_importance = Column(Text, nullable=True)  # JSON string
    
    # Relationships
    dataset = relationship("Dataset", back_populates="forecasts")
    forecast_data = relationship("ForecastData", back_populates="forecast", cascade="all, delete-orphan")

class ForecastData(Base):
    """Table for storing detailed forecast data points"""
    __tablename__ = 'forecast_data'
    
    id = Column(Integer, primary_key=True)
    forecast_id = Column(Integer, ForeignKey('forecasts.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    predicted = Column(Float, nullable=False)
    actual = Column(Float, nullable=True)  # Null for future predictions
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    
    # Relationships
    forecast = relationship("Forecast", back_populates="forecast_data")

# Database connection
def get_engine():
    """Create and return a database engine using environment variables"""
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    return create_engine(database_url)

def init_db():
    """Initialize the database by creating all tables"""
    engine = get_engine()
    Base.metadata.create_all(engine)
    return engine

def get_session():
    """Create and return a new database session"""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

# Dataset Operations
def save_dataset(data, name, description=None):
    """
    Save a dataset to the database
    
    Args:
        data (pd.DataFrame): DataFrame with price data
        name (str): Name of the dataset
        description (str, optional): Description of the dataset
    
    Returns:
        int: ID of the created dataset
    """
    # Make sure timestamps are datetime objects
    if not pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    session = get_session()
    
    try:
        # Create the dataset record
        has_external = any(col in data.columns for col in 
                          ['temperature', 'wind_speed', 'solar_irradiance', 
                           'precipitation', 'load'])
        
        dataset = Dataset(
            name=name,
            description=description,
            start_date=data['timestamp'].min(),
            end_date=data['timestamp'].max(),
            record_count=len(data),
            has_external_features=has_external
        )
        
        session.add(dataset)
        session.flush()  # Get the ID
        
        # Insert the price data
        for _, row in data.iterrows():
            price_data = PriceData(
                dataset_id=dataset.id,
                timestamp=row['timestamp'],
                price=row['price'],
                temperature=row.get('temperature'),
                wind_speed=row.get('wind_speed'),
                solar_irradiance=row.get('solar_irradiance'),
                precipitation=row.get('precipitation'),
                load=row.get('load')
            )
            session.add(price_data)
        
        session.commit()
        return dataset.id
    
    except Exception as e:
        session.rollback()
        raise e
    
    finally:
        session.close()

def get_all_datasets():
    """
    Get all available datasets
    
    Returns:
        list: List of dataset records
    """
    session = get_session()
    try:
        return session.query(Dataset).all()
    finally:
        session.close()

def get_dataset_by_id(dataset_id):
    """
    Get a dataset by ID
    
    Args:
        dataset_id (int): ID of the dataset
    
    Returns:
        Dataset: Dataset record
    """
    session = get_session()
    try:
        return session.query(Dataset).filter(Dataset.id == dataset_id).first()
    finally:
        session.close()

def get_dataset_data(dataset_id):
    """
    Get the price data for a dataset
    
    Args:
        dataset_id (int): ID of the dataset
    
    Returns:
        pd.DataFrame: DataFrame with the price data
    """
    session = get_session()
    try:
        # Get all price data for the dataset
        price_data = session.query(PriceData).filter(
            PriceData.dataset_id == dataset_id
        ).all()
        
        # Convert to DataFrame
        data = []
        for record in price_data:
            row = {
                'timestamp': record.timestamp,
                'price': record.price
            }
            
            # Add optional columns if they have values
            for col in ['temperature', 'wind_speed', 'solar_irradiance', 
                        'precipitation', 'load']:
                value = getattr(record, col)
                if value is not None:
                    row[col] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    finally:
        session.close()

# Forecast Operations
def save_forecast(dataset_id, forecast_df, error_metrics, feature_importance, 
                 name, forecast_horizon, confidence_interval, models_used):
    """
    Save a forecast to the database
    
    Args:
        dataset_id (int): ID of the dataset
        forecast_df (pd.DataFrame): DataFrame with forecast data
        error_metrics (dict): Dictionary of error metrics
        feature_importance (dict): Dictionary of feature importance
        name (str): Name of the forecast
        forecast_horizon (int): Forecast horizon in days
        confidence_interval (int): Confidence interval percentage
        models_used (list): List of models used for the forecast
    
    Returns:
        int: ID of the created forecast
    """
    session = get_session()
    
    try:
        # Create the forecast record
        forecast = Forecast(
            dataset_id=dataset_id,
            name=name,
            forecast_horizon=forecast_horizon,
            confidence_interval=confidence_interval,
            models_used=','.join(models_used),
            error_metrics=json.dumps(error_metrics) if error_metrics else None,
            feature_importance=json.dumps(feature_importance) if feature_importance else None
        )
        
        session.add(forecast)
        session.flush()  # Get the ID
        
        # Insert the forecast data
        for _, row in forecast_df.iterrows():
            # Convert timestamp if it's not already a datetime
            timestamp = row['timestamp']
            if not isinstance(timestamp, datetime.datetime):
                timestamp = pd.to_datetime(timestamp)
            
            forecast_data = ForecastData(
                forecast_id=forecast.id,
                timestamp=timestamp,
                predicted=row['predicted'],
                actual=row.get('actual'),  # Might be NaN for future predictions
                lower_bound=row.get('lower'),
                upper_bound=row.get('upper')
            )
            session.add(forecast_data)
        
        session.commit()
        return forecast.id
    
    except Exception as e:
        session.rollback()
        raise e
    
    finally:
        session.close()

def get_all_forecasts(dataset_id=None):
    """
    Get all available forecasts
    
    Args:
        dataset_id (int, optional): Filter by dataset ID
    
    Returns:
        list: List of forecast records
    """
    session = get_session()
    try:
        query = session.query(Forecast)
        if dataset_id is not None:
            query = query.filter(Forecast.dataset_id == dataset_id)
        return query.all()
    finally:
        session.close()

def get_forecast_by_id(forecast_id):
    """
    Get a forecast by ID
    
    Args:
        forecast_id (int): ID of the forecast
    
    Returns:
        Forecast: Forecast record
    """
    session = get_session()
    try:
        return session.query(Forecast).filter(Forecast.id == forecast_id).first()
    finally:
        session.close()

def get_forecast_data(forecast_id):
    """
    Get the forecast data for a forecast
    
    Args:
        forecast_id (int): ID of the forecast
    
    Returns:
        tuple: (forecast_df, error_metrics, feature_importance)
    """
    session = get_session()
    try:
        # Get the forecast record
        forecast = session.query(Forecast).filter(Forecast.id == forecast_id).first()
        
        if not forecast:
            return None, None, None
        
        # Get all forecast data points
        forecast_data = session.query(ForecastData).filter(
            ForecastData.forecast_id == forecast_id
        ).all()
        
        # Convert to DataFrame
        data = []
        for record in forecast_data:
            row = {
                'timestamp': record.timestamp,
                'predicted': record.predicted,
                'actual': record.actual
            }
            
            # Add confidence intervals if they exist
            if record.lower_bound is not None:
                row['lower'] = record.lower_bound
            if record.upper_bound is not None:
                row['upper'] = record.upper_bound
            
            data.append(row)
        
        forecast_df = pd.DataFrame(data)
        
        # Parse error metrics and feature importance
        error_metrics = json.loads(forecast.error_metrics) if forecast.error_metrics else None
        feature_importance = json.loads(forecast.feature_importance) if forecast.feature_importance else None
        
        return forecast_df, error_metrics, feature_importance
    
    finally:
        session.close()

def delete_dataset(dataset_id):
    """
    Delete a dataset and all related data
    
    Args:
        dataset_id (int): ID of the dataset
    
    Returns:
        bool: True if successful
    """
    session = get_session()
    try:
        dataset = session.query(Dataset).filter(Dataset.id == dataset_id).first()
        if dataset:
            session.delete(dataset)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def delete_forecast(forecast_id):
    """
    Delete a forecast and all related data
    
    Args:
        forecast_id (int): ID of the forecast
    
    Returns:
        bool: True if successful
    """
    session = get_session()
    try:
        forecast = session.query(Forecast).filter(Forecast.id == forecast_id).first()
        if forecast:
            session.delete(forecast)
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

# Initialize database tables if they don't exist
def initialize():
    """Initialize the database if it doesn't exist"""
    try:
        init_db()
        return True
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False