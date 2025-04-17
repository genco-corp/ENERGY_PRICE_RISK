#!/bin/bash

# Install required dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set the environment variable (in case .env file is not loaded)
export DATABASE_URL=sqlite:///electricity_forecasting.db

# Initialize the database
echo "Initializing database..."
python init_db.py

# Run the application
echo "Starting the application..."
streamlit run app.py 