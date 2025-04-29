import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

class BangladeshDataProcessor:
    def __init__(self):
        self.weather_data = None
        self.crop_data = None
        self.climate_data = None
        self.agricultural_data = None
        self.scaler = MinMaxScaler()
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'rainfall',
            'avg_temperature', 'annual_rainfall', 'aqi',
            'forest_cover', 'drought_severity', 'agricultural_yield'
        ]
        
    def load_all_data(self):
        try:
            # Load weather data
            self.weather_data = pd.read_csv('data/bangladesh/weather_data.csv')
            self.weather_data['date'] = pd.to_datetime(self.weather_data['date'])
            
            # Load climate impact data
            self.climate_data = pd.read_csv('data/bangladesh/Bangladesh_Environmental_Climate_Change_Impact.csv')
            self.climate_data['District'] = self.climate_data['District'].str.strip()
            
            # Load crop data
            self.crop_data = pd.read_csv('data/bangladesh/crop_data.csv')
            self.crop_data['date'] = pd.to_datetime(self.crop_data['date'])
            
            # Load agricultural data if available
            try:
                self.agricultural_data = pd.read_csv('data/bangladesh/Agricultural Dataset.csv')
            except:
                print("Agricultural dataset not found, continuing without it.")
            
            return self.combine_datasets()
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise
    
    def combine_datasets(self):
        try:
            # Start with weather data as base
            combined_data = self.weather_data.copy()
            
            # Add crop yield data
            if self.crop_data is not None:
                combined_data = pd.merge(
                    combined_data,
                    self.crop_data,
                    on=['date', 'district'],
                    how='left'
                )
            
            # Extract year from date for merging with climate data
            combined_data['Year'] = combined_data['date'].dt.year
            
            # Add climate impact features
            if self.climate_data is not None:
                # Rename district column to match
                climate_features = [
                    'Avg_Temperature_C', 'Annual_Rainfall_mm', 'AQI',
                    'Forest_Cover_Percent', 'Drought_Severity',
                    'Agricultural_Yield_ton_per_hectare'
                ]
                
                climate_subset = self.climate_data[['Year', 'District'] + climate_features].copy()
                climate_subset = climate_subset.rename(columns={'District': 'district'})
                
                combined_data = pd.merge(
                    combined_data,
                    climate_subset,
                    on=['Year', 'district'],
                    how='left'
                )
            
            # Fill missing values with appropriate methods
            numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
            combined_data[numeric_columns] = combined_data[numeric_columns].fillna(combined_data[numeric_columns].mean())
            
            # Only keep features that actually exist in the dataset
            self.feature_columns = [col for col in self.feature_columns if col in combined_data.columns]
            
            return combined_data
            
        except Exception as e:
            print(f"Error combining datasets: {str(e)}")
            raise
    
    def preprocess_data(self, data):
        try:
            # Ensure all feature columns exist
            missing_cols = [col for col in self.feature_columns if col not in data.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                # Remove missing columns from feature list
                self.feature_columns = [col for col in self.feature_columns if col in data.columns]
            
            if not self.feature_columns:
                raise ValueError("No valid feature columns available for preprocessing")
            
            # Scale features
            features = data[self.feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            return scaled_features
            
        except Exception as e:
            print(f"Error preprocessing data: {str(e)}")
            raise
    
    def get_district_coordinates(self):
        # Return unique districts with their coordinates
        districts = self.weather_data[['district', 'latitude', 'longitude']].drop_duplicates()
        return districts.to_dict('records')
    
    def get_district_info(self, district_name):
        try:
            # Get the most recent climate data for the district
            district_data = self.climate_data[
                self.climate_data['District'].str.lower() == district_name.lower()
            ].sort_values('Year', ascending=False).iloc[0]
            
            return {
                'avg_temperature': float(district_data.get('Avg_Temperature_C', 0)),
                'annual_rainfall': float(district_data.get('Annual_Rainfall_mm', 0)),
                'aqi': float(district_data.get('AQI', 0)),
                'forest_cover': float(district_data.get('Forest_Cover_Percent', 0)),
                'drought_severity': float(district_data.get('Drought_Severity', 0)),
                'agricultural_yield': float(district_data.get('Agricultural_Yield_ton_per_hectare', 0))
            }
        except Exception as e:
            print(f"Error getting district info: {str(e)}")
            return {
                'avg_temperature': 0,
                'annual_rainfall': 0,
                'aqi': 0,
                'forest_cover': 0,
                'drought_severity': 0,
                'agricultural_yield': 0
            }
    
    def inverse_transform_prediction(self, scaled_prediction):
        return self.scaler.inverse_transform(scaled_prediction.reshape(-1, 1)) 