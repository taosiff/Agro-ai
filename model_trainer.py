import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import yaml
import os
from datetime import datetime, timedelta
import joblib

class AgroClimateModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        self.data_paths = {
            'weather': 'data/bangladesh/weather_data.csv',
            'climate': 'data/bangladesh/Bangladesh_Environmental_Climate_Change_Impact.csv',
            'crop': 'data/bangladesh/crop_data.csv',
            'agricultural': 'data/bangladesh/Agricultural Dataset.csv',
            'temp_rain': 'data/bangladesh/Temp_and_rain.csv',
            'earthquake': 'data/bangladesh/clead_earth_quake_data.csv'
        }
        
    def load_and_preprocess_data(self):
        """Load and preprocess all datasets"""
        processed_data = {}
        
        # Load weather data
        try:
            weather_data = pd.read_csv(self.data_paths['weather'])
            weather_data['date'] = pd.to_datetime(weather_data['date'])
            weather_data['year'] = weather_data['date'].dt.year
            processed_data['weather'] = weather_data
        except Exception as e:
            print(f"Error loading weather data: {e}")
            processed_data['weather'] = pd.DataFrame()
        
        # Load climate data
        try:
            climate_data = pd.read_csv(self.data_paths['climate'])
            climate_data['Year'] = pd.to_numeric(climate_data['Year'], errors='coerce')
            processed_data['climate'] = climate_data
        except Exception as e:
            print(f"Error loading climate data: {e}")
            processed_data['climate'] = pd.DataFrame()
        
        # Merge weather and climate data
        if not processed_data['weather'].empty and not processed_data['climate'].empty:
            merged_data = pd.merge(
                processed_data['weather'],
                processed_data['climate'],
                left_on=['year', 'district'],
                right_on=['Year', 'District'],
                how='left'
            )
            
            # Fill missing values with mean values
            numeric_columns = merged_data.select_dtypes(include=[np.number]).columns
            merged_data[numeric_columns] = merged_data[numeric_columns].fillna(merged_data[numeric_columns].mean())
            
            processed_data['merged'] = merged_data
        else:
            processed_data['merged'] = pd.DataFrame()
        
        return processed_data
    
    def train_rainfall_model(self, data):
        """Train model to predict rainfall"""
        if data['merged'].empty:
            print("No data available for training rainfall model")
            return 0
            
        # Prepare features for rainfall prediction
        features = ['temperature', 'humidity', 'wind_speed', 'Avg_Temperature_C', 'Annual_Rainfall_mm']
        target = 'rainfall'
        
        # Filter available features
        available_features = [f for f in features if f in data['merged'].columns]
        
        if not available_features:
            print("No suitable features available for rainfall prediction")
            return 0
        
        # Create training data
        X = data['merged'][available_features].fillna(0)
        y = data['merged'][target].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        self.models['rainfall'] = model
        self.scalers['rainfall'] = scaler
        self.feature_columns['rainfall'] = available_features
        
        return model.score(X_test_scaled, y_test)
    
    def train_flood_model(self, data):
        """Train model to predict flood risk"""
        if data['merged'].empty:
            print("No data available for training flood model")
            return 0
            
        # Prepare features for flood prediction
        features = ['rainfall', 'temperature', 'humidity', 'wind_speed', 'River_Water_Level_m', 'Annual_Rainfall_mm']
        
        # Filter available features
        available_features = [f for f in features if f in data['merged'].columns]
        
        if not available_features:
            print("No suitable features available for flood prediction")
            return 0
        
        # Create flood risk target (binary classification based on rainfall and water level)
        data['merged']['flood_risk'] = ((data['merged']['rainfall'] > data['merged']['rainfall'].mean()) & 
                                      (data['merged']['River_Water_Level_m'] > data['merged']['River_Water_Level_m'].mean())).astype(int)
        
        # Create training data
        X = data['merged'][available_features].fillna(0)
        y = data['merged']['flood_risk']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        self.models['flood'] = model
        self.scalers['flood'] = scaler
        self.feature_columns['flood'] = available_features
        
        return model.score(X_test_scaled, y_test)
    
    def train_crop_yield_model(self, data):
        """Train model to predict crop yield"""
        if data['merged'].empty:
            print("No data available for training crop yield model")
            return 0
            
        # Prepare features for crop yield prediction
        features = ['temperature', 'rainfall', 'humidity', 'wind_speed', 'Avg_Temperature_C', 
                   'Annual_Rainfall_mm', 'AQI', 'Forest_Cover_Percent', 'Drought_Severity']
        target = 'Agricultural_Yield_ton_per_hectare'
        
        # Filter available features
        available_features = [f for f in features if f in data['merged'].columns]
        
        if not available_features or target not in data['merged'].columns:
            print("No suitable features or target available for crop yield prediction")
            return 0
        
        # Create training data
        X = data['merged'][available_features].fillna(0)
        y = data['merged'][target].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        self.models['crop_yield'] = model
        self.scalers['crop_yield'] = scaler
        self.feature_columns['crop_yield'] = available_features
        
        return model.score(X_test_scaled, y_test)
    
    def save_models(self, filename='models.yml'):
        """Save all trained models and their configurations to a YAML file"""
        model_data = {
            'models': {},
            'scalers': {},
            'feature_columns': self.feature_columns
        }
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save each model
        for name, model in self.models.items():
            model_path = f'models/{name}_model.joblib'
            joblib.dump(model, model_path)
            model_data['models'][name] = model_path
            
            # Save corresponding scaler
            scaler_path = f'models/{name}_scaler.joblib'
            joblib.dump(self.scalers[name], scaler_path)
            model_data['scalers'][name] = scaler_path
        
        # Save to YAML
        with open(filename, 'w') as f:
            yaml.dump(model_data, f)
    
    def train_all_models(self):
        """Train all models and save them"""
        print("Loading and preprocessing data...")
        data = self.load_and_preprocess_data()
        
        print("\nTraining rainfall prediction model...")
        rainfall_score = self.train_rainfall_model(data)
        print(f"Rainfall model R2 score: {rainfall_score:.4f}")
        
        print("\nTraining flood risk model...")
        flood_score = self.train_flood_model(data)
        print(f"Flood model accuracy: {flood_score:.4f}")
        
        print("\nTraining crop yield model...")
        crop_score = self.train_crop_yield_model(data)
        print(f"Crop yield model R2 score: {crop_score:.4f}")
        
        print("\nTraining demand forecasting model...")
        demand_score = self.train_demand_forecast_model(data)
        print(f"Demand forecast model R2 score: {demand_score:.4f}")
        
        print("\nSaving models...")
        self.save_models()
        print("Models saved successfully!")

    def train_demand_forecast_model(self, data):
        """Train model to predict product demand based on weather, social trends, and historical sales"""
        if data['merged'].empty:
            print("No data available for training demand forecast model")
            return 0
            
        # Prepare features for demand prediction
        features = [
            'temperature', 'rainfall', 'humidity', 'wind_speed',
            'Avg_Temperature_C', 'Annual_Rainfall_mm', 'AQI',
            'Forest_Cover_Percent', 'Drought_Severity',
            'Agricultural_Yield_ton_per_hectare'
        ]
        
        # Add time-based features
        data['merged']['month'] = data['merged']['date'].dt.month
        data['merged']['day_of_week'] = data['merged']['date'].dt.dayofweek
        data['merged']['is_weekend'] = data['merged']['day_of_week'].isin([5, 6]).astype(int)
        
        # Add seasonal features
        data['merged']['is_rainy_season'] = data['merged']['month'].isin([6, 7, 8, 9]).astype(int)
        data['merged']['is_harvest_season'] = data['merged']['month'].isin([3, 4, 10, 11]).astype(int)
        
        # Add these new features to the feature list
        features.extend(['month', 'day_of_week', 'is_weekend', 'is_rainy_season', 'is_harvest_season'])
        
        # Filter available features
        available_features = [f for f in features if f in data['merged'].columns]
        
        # For demonstration, we'll use crop yield as a proxy for demand
        # In a real system, this would be replaced with actual sales data
        target = 'Agricultural_Yield_ton_per_hectare'
        
        if not available_features or target not in data['merged'].columns:
            print("No suitable features or target available for demand forecasting")
            return 0
        
        # Create training data
        X = data['merged'][available_features].fillna(0)
        y = data['merged'][target].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model with more trees and deeper depth for complex patterns
        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        self.models['demand_forecast'] = model
        self.scalers['demand_forecast'] = scaler
        self.feature_columns['demand_forecast'] = available_features
        
        return model.score(X_test_scaled, y_test)

if __name__ == '__main__':
    trainer = AgroClimateModelTrainer()
    trainer.train_all_models() 