import yaml
import joblib
import numpy as np
from datetime import datetime, timedelta
import openai
from model_trainer import AgroClimateModelTrainer
import logging
import re
import os
import requests
from typing import Dict, Any, List
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AgroClimateAI:
    def __init__(self, model_config_path='models.yml'):
        try:
            # Load model configuration
            with open(model_config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Load models and scalers
            self.models = {}
            self.scalers = {}
            for name, path in self.config['models'].items():
                self.models[name] = joblib.load(path)
                self.scalers[name] = joblib.load(self.config['scalers'][name])
            
            # Initialize model trainer for data access
            self.trainer = AgroClimateModelTrainer()
            self.data = self.trainer.load_and_preprocess_data()
            
            # Define conversation context
            self.conversation_context = {
                'current_topic': None,
                'previous_queries': [],
                'district': None,
                'last_prediction': None
            }
            
            # Check for API keys
            self.openai_api_key = os.getenv('OPENAI_API_KEY')
            self.weather_api_key = os.getenv('WEATHER_API_KEY')
            
            if not self.openai_api_key:
                logger.warning("OpenAI API key not found in environment variables")
            else:
                logger.info("OpenAI API key loaded successfully")
            
            if not self.weather_api_key:
                logger.warning("Weather API key not found in environment variables")
            else:
                logger.info("Weather API key loaded successfully")
            
            logger.info("AgroClimateAI initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AgroClimateAI: {str(e)}")
            raise

    def get_district_data(self, district: str) -> Dict[str, Any]:
        """Get the most recent data for a district"""
        try:
            # District coordinates for Bangladesh
            district_coordinates = {
                'dhaka': {'latitude': 23.8103, 'longitude': 90.4125},
                'chittagong': {'latitude': 22.3569, 'longitude': 91.7832},
                'khulna': {'latitude': 22.8456, 'longitude': 89.5403},
                'rajshahi': {'latitude': 24.3740, 'longitude': 88.6011},
                'sylhet': {'latitude': 24.8949, 'longitude': 91.8687},
                'barisal': {'latitude': 22.7010, 'longitude': 90.3535},
                'rangpur': {'latitude': 25.7439, 'longitude': 89.2752},
                'mymensingh': {'latitude': 24.7471, 'longitude': 90.4203}
            }
            
            # Get coordinates for the district
            district_lower = district.lower()
            if district_lower in district_coordinates:
                return district_coordinates[district_lower]
            
            # If district not found, use Dhaka as default
            logger.warning(f"Coordinates not found for district: {district}, using Dhaka coordinates")
            return district_coordinates['dhaka']
            
        except Exception as e:
            logger.error(f"Error getting district data: {str(e)}")
            return {}

    def get_real_time_weather(self, district: str) -> Dict[str, Any]:
        """Fetch real-time weather data for a district"""
        try:
            if not self.weather_api_key:
                logger.error("Weather API key not found")
                return {}
            
            # Get district coordinates
            district_data = self.get_district_data(district)
            if not district_data:
                return {}
            
            lat = district_data.get('latitude')
            lon = district_data.get('longitude')
            
            if not lat or not lon:
                return {}
            
            # Fetch current weather using WeatherAPI
            url = f"http://api.weatherapi.com/v1/current.json?key={self.weather_api_key}&q={lat},{lon}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Weather API error: {response.status_code}")
                return {}
            
            weather_data = response.json()
            current = weather_data['current']
            
            return {
                'temperature': current['temp_c'],
                'humidity': current['humidity'],
                'wind_speed': current['wind_kph'],
                'rainfall': current.get('precip_mm', 0),
                'description': current['condition']['text'],
                'feels_like': current['feelslike_c'],
                'pressure': current['pressure_mb'],
                'visibility': current['vis_km']
            }
        except Exception as e:
            logger.error(f"Error fetching real-time weather: {str(e)}")
            return {}

    def get_weather_forecast(self, district: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get weather forecast for a district"""
        try:
            if not self.weather_api_key:
                logger.error("Weather API key not found")
                return []
            
            # Get district coordinates
            district_data = self.get_district_data(district)
            if not district_data:
                return []
            
            lat = district_data.get('latitude')
            lon = district_data.get('longitude')
            
            if not lat or not lon:
                return []
            
            # Fetch forecast using WeatherAPI
            url = f"http://api.weatherapi.com/v1/forecast.json?key={self.weather_api_key}&q={lat},{lon}&days={days}"
            response = requests.get(url)
            
            if response.status_code != 200:
                logger.error(f"Weather API error: {response.status_code}")
                return []
            
            forecast_data = response.json()
            forecasts = []
            
            for day in forecast_data['forecast']['forecastday']:
                for hour in day['hour']:
                    forecast = {
                        'date': hour['time'],
                        'temperature': hour['temp_c'],
                        'humidity': hour['humidity'],
                        'wind_speed': hour['wind_kph'],
                        'rainfall': hour.get('precip_mm', 0),
                        'description': hour['condition']['text']
                    }
                    forecasts.append(forecast)
            
            return forecasts
        except Exception as e:
            logger.error(f"Error getting weather forecast: {str(e)}")
            return []

    def predict_flood_risk(self, district: str) -> Dict[str, Any]:
        """Predict flood risk based on weather data and historical patterns"""
        try:
            # Get current weather and forecast
            current_weather = self.get_real_time_weather(district)
            forecast = self.get_weather_forecast(district, days=7)
            
            if not current_weather or not forecast:
                return {
                    'risk_level': 'Unknown',
                    'probability': 0,
                    'recommendation': 'Unable to predict flood risk at this time.'
                }
            
            # Calculate risk factors
            rainfall_risk = 0
            if current_weather.get('rainfall', 0) > 50:  # Heavy rainfall threshold
                rainfall_risk += 0.4
            
            # Check forecast for heavy rainfall
            heavy_rain_days = sum(1 for item in forecast if item.get('rainfall', 0) > 50)
            rainfall_risk += min(heavy_rain_days * 0.1, 0.3)  # Up to 30% additional risk
            
            # Calculate total risk probability
            total_risk = min(rainfall_risk, 1.0)  # Cap at 100%
            
            # Determine risk level
            if total_risk < 0.2:
                risk_level = 'Low'
                recommendation = 'Normal farming activities can continue.'
            elif total_risk < 0.5:
                risk_level = 'Moderate'
                recommendation = 'Monitor weather closely. Consider harvesting early if crops are ready.'
            elif total_risk < 0.8:
                risk_level = 'High'
                recommendation = 'High risk of flooding. Consider immediate harvest if crops are ready.'
            else:
                risk_level = 'Very High'
                recommendation = 'Immediate action recommended. Harvest crops if possible.'
            
            return {
                'risk_level': risk_level,
                'probability': round(total_risk * 100, 1),
                'recommendation': recommendation,
                'factors': {
                    'current_rainfall': current_weather.get('rainfall', 0),
                    'heavy_rain_days': heavy_rain_days
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting flood risk: {str(e)}")
            return {
                'risk_level': 'Error',
                'probability': 0,
                'recommendation': 'Unable to predict flood risk at this time.'
            }

    def predict_demand(self, district: str, sku: str = None) -> Dict[str, Any]:
        """Predict demand for agricultural products based on weather, social trends, and historical data"""
        try:
            # Get current weather and forecast
            current_weather = self.get_real_time_weather(district)
            forecast = self.get_weather_forecast(district, days=7)
            
            if not current_weather or not forecast:
                return {
                    'error': 'Unable to fetch weather data for demand prediction'
                }
            
            # Get district data
            district_data = self.get_district_data(district)
            if not district_data:
                return {
                    'error': f'No data available for district: {district}'
                }
            
            # Prepare features for prediction
            features = {}
            
            # Add current weather features
            features.update({
                'temperature': current_weather.get('temperature', 0),
                'rainfall': current_weather.get('rainfall', 0),
                'humidity': current_weather.get('humidity', 0),
                'wind_speed': current_weather.get('wind_speed', 0)
            })
            
            # Add district-level features
            features.update({
                'Avg_Temperature_C': district_data.get('avg_temperature', 0),
                'Annual_Rainfall_mm': district_data.get('annual_rainfall', 0),
                'AQI': district_data.get('aqi', 0),
                'Forest_Cover_Percent': district_data.get('forest_cover', 0),
                'Drought_Severity': district_data.get('drought_severity', 0),
                'Agricultural_Yield_ton_per_hectare': district_data.get('agricultural_yield', 0)
            })
            
            # Add time-based features
            current_date = datetime.now()
            features.update({
                'month': current_date.month,
                'day_of_week': current_date.weekday(),
                'is_weekend': 1 if current_date.weekday() in [5, 6] else 0,
                'is_rainy_season': 1 if current_date.month in [6, 7, 8, 9] else 0,
                'is_harvest_season': 1 if current_date.month in [3, 4, 10, 11] else 0
            })
            
            # Convert features to DataFrame
            X = pd.DataFrame([features])
            
            # Scale features
            X_scaled = self.scalers['demand_forecast'].transform(X)
            
            # Make prediction
            predicted_demand = self.models['demand_forecast'].predict(X_scaled)[0]
            
            # Calculate confidence interval (assuming normal distribution)
            std_dev = np.std(self.models['demand_forecast'].predict(X_scaled))
            confidence_interval = {
                'lower': max(0, predicted_demand - 1.96 * std_dev),
                'upper': predicted_demand + 1.96 * std_dev
            }
            
            # Generate recommendations based on prediction
            if predicted_demand > confidence_interval['upper']:
                recommendation = "High demand expected. Consider increasing inventory."
            elif predicted_demand < confidence_interval['lower']:
                recommendation = "Low demand expected. Consider reducing inventory."
            else:
                recommendation = "Demand is within expected range. Maintain current inventory levels."
            
            return {
                'predicted_demand': round(predicted_demand, 2),
                'confidence_interval': {
                    'lower': round(confidence_interval['lower'], 2),
                    'upper': round(confidence_interval['upper'], 2)
                },
                'recommendation': recommendation,
                'factors': {
                    'weather_impact': {
                        'temperature': current_weather.get('temperature', 0),
                        'rainfall': current_weather.get('rainfall', 0)
                    },
                    'seasonal_factors': {
                        'is_rainy_season': features['is_rainy_season'],
                        'is_harvest_season': features['is_harvest_season']
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting demand: {str(e)}")
            return {
                'error': 'Unable to predict demand at this time.'
            }

    def process_query(self, query: str, district: str) -> str:
        """Process natural language query with enhanced NLP understanding"""
        try:
            # Update conversation context
            self.conversation_context['district'] = district
            self.conversation_context['previous_queries'].append(query)
            
            # Handle basic interactions
            if query.lower() in ['hi', 'hello', 'hey']:
                return f"Hello! I'm your AgroClimateAI assistant for {district}. What would you like to know?"
            
            if query.lower() in ['how are you', 'how are you doing']:
                return "I'm doing well. How can I help you with agricultural information for " + district + "?"
            
            # Check for flood-related queries
            if any(word in query.lower() for word in ['flood', 'water', 'risk', 'danger']):
                flood_risk = self.predict_flood_risk(district)
                response = f"Flood Risk Assessment for {district}:\n\n"
                response += f"Risk Level: {flood_risk['risk_level']}\n"
                response += f"Probability: {flood_risk['probability']}%\n\n"
                response += f"Recommendation: {flood_risk['recommendation']}\n\n"
                response += "Factors considered:\n"
                response += f"• Current Rainfall: {flood_risk['factors']['current_rainfall']} mm\n"
                response += f"• Heavy Rain Days Expected: {flood_risk['factors']['heavy_rain_days']}\n"
                return response
            
            # Check for specific forecast requests
            if any(word in query.lower() for word in ['forecast', '7 days', 'week', 'next week']):
                forecast = self.get_weather_forecast(district, days=7)
                if forecast:
                    response = f"Weather forecast for {district} for the next 7 days:\n\n"
                    current_date = None
                    for item in forecast:
                        date = datetime.strptime(item['date'], '%Y-%m-%d %H:%M').strftime('%Y-%m-%d')
                        if date != current_date:
                            current_date = date
                            response += f"\n{date}:\n"
                        response += f"• Time: {datetime.strptime(item['date'], '%Y-%m-%d %H:%M').strftime('%H:%M')}\n"
                        response += f"• Temperature: {item['temperature']}°C\n"
                        response += f"• Humidity: {item['humidity']}%\n"
                        response += f"• Wind Speed: {item['wind_speed']} km/h\n"
                        response += f"• Rainfall: {item['rainfall']} mm\n"
                        response += f"• Conditions: {item['description']}\n"
                    return response
                else:
                    return f"Sorry, I couldn't fetch the 7-day forecast for {district} at this moment."
            
            # Get real-time weather data
            current_weather = self.get_real_time_weather(district)
            
            # Prepare context for OpenAI
            context = f"""
            Current Weather in {district}:
            - Temperature: {current_weather.get('temperature', 'N/A')}°C
            - Feels Like: {current_weather.get('feels_like', 'N/A')}°C
            - Humidity: {current_weather.get('humidity', 'N/A')}%
            - Wind Speed: {current_weather.get('wind_speed', 'N/A')} km/h
            - Pressure: {current_weather.get('pressure', 'N/A')} hPa
            - Visibility: {current_weather.get('visibility', 'N/A')} km
            - Conditions: {current_weather.get('description', 'N/A')}
            """
            
            # Generate response using OpenAI
            try:
                if not self.openai_api_key:
                    logger.error("OpenAI API key not found")
                    return self._generate_fallback_response(district, current_weather, [])
                
                # Configure OpenAI API
                client = openai.OpenAI(api_key=self.openai_api_key)
                
                # Generate a direct response
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system",
                            "content": """You are an agricultural expert assistant for Bangladesh. 
                            Provide direct, concise answers to the user's specific questions.
                            Only answer what is being asked.
                            Keep responses brief and to the point.
                            Do not provide additional context or recommendations unless specifically asked."""
                        },
                        {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                    ],
                    temperature=0.7,
                    max_tokens=200
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                logger.error(f"OpenAI API error: {str(e)}")
                return self._generate_fallback_response(district, current_weather, [])
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._generate_fallback_response(district, {}, [])

    def _generate_fallback_response(self, district: str, current_weather: Dict[str, Any], forecast: List[Dict[str, Any]]) -> str:
        """Generate a fallback response when OpenAI API is unavailable"""
        try:
            response = f"Current Weather in {district}:\n"
            
            # Add current weather data
            response += f"• Temperature: {current_weather.get('temperature', 'N/A')}°C\n"
            response += f"• Feels Like: {current_weather.get('feels_like', 'N/A')}°C\n"
            response += f"• Humidity: {current_weather.get('humidity', 'N/A')}%\n"
            response += f"• Wind Speed: {current_weather.get('wind_speed', 'N/A')} m/s\n"
            response += f"• Pressure: {current_weather.get('pressure', 'N/A')} hPa\n"
            response += f"• Visibility: {current_weather.get('visibility', 'N/A')} km\n"
            response += f"• Conditions: {current_weather.get('description', 'N/A')}\n"
            
            if forecast:
                response += "\nNext 24 Hours:\n"
                for item in forecast[:4]:
                    response += f"• {item['date']}: {item['temperature']}°C, {item['description']}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating fallback response: {str(e)}")
            return f"I can help you with agricultural information for {district}. What would you like to know?" 