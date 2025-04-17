#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variable
export DATABASE_URL=sqlite:///electricity_forecasting.db

# Initialize database
echo "Initializing database..."
python init_db.py

# Run application on port 8501
echo "Starting application on port 8501..."
streamlit run app.py --server.port 8501 