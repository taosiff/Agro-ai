import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
from sklearn.ensemble import RandomForestRegressor
from data_processor import BangladeshDataProcessor
from openai import OpenAI
from agro_ai import AgroClimateAI
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Get OpenAI API key
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# Initialize the AgroClimateAI system
try:
    agro_ai = AgroClimateAI()
    logger.info("AgroClimateAI system initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize AgroClimateAI system: {str(e)}")
    raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'response': 'No data received'})
            
        query = data.get('query', '').strip()
        district = data.get('district', '').strip()
        
        logger.info(f"Received chat request - Query: {query}, District: {district}")
        
        if not query:
            return jsonify({'response': 'Please enter a question.'})
        
        if not district:
            return jsonify({'response': 'Please select a district first.'})
        
        # Process the query using AgroClimateAI
        response = agro_ai.process_query(query, district)
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({'response': 'Sorry, I encountered an error. Please try again.'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'})
            
        district = data.get('district', '').strip()
        
        if not district:
            return jsonify({'error': 'Please select a district.'})
        
        # Get predictions
        rainfall = agro_ai.predict_rainfall(district)
        flood_risk = agro_ai.predict_flood_risk(district)
        forecast = agro_ai.get_weather_forecast(district)
        
        return jsonify({
            'rainfall': rainfall,
            'flood_risk': flood_risk,
            'forecast': forecast
        })
        
    except Exception as e:
        logger.error(f"Error making predictions: {str(e)}")
        return jsonify({'error': 'Failed to make predictions.'})

@app.route('/predict/demand', methods=['POST'])
def predict_demand():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'})
            
        district = data.get('district', '').strip()
        sku = data.get('sku', '').strip()  # Optional SKU parameter
        
        if not district:
            return jsonify({'error': 'Please select a district.'})
        
        # Get demand prediction
        demand_prediction = agro_ai.predict_demand(district, sku)
        
        return jsonify(demand_prediction)
        
    except Exception as e:
        logger.error(f"Error predicting demand: {str(e)}")
        return jsonify({'error': 'Failed to predict demand.'})

if __name__ == '__main__':
    app.run(debug=True) 