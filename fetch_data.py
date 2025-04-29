import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import time

class BangladeshDataFetcher:
    def __init__(self, data_dir='data/bangladesh'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # NASA POWER API endpoint
        self.power_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
        # Major districts in Bangladesh with their coordinates
        self.districts = {
            'Dhaka': (23.8103, 90.4125),
            'Chittagong': (22.3569, 91.7832),
            'Khulna': (22.8456, 89.5403),
            'Rajshahi': (24.3740, 88.6011),
            'Sylhet': (24.8949, 91.8687),
            'Barisal': (22.7010, 90.3535),
            'Rangpur': (25.7439, 89.2752),
            'Mymensingh': (24.7471, 90.4203)
        }
        
        # Parameters to fetch from NASA POWER
        self.parameters = [
            'T2M',  # Temperature at 2m
            'RH2M',  # Relative Humidity at 2m
            'PRECTOTCORR',  # Precipitation
            'WS2M'  # Wind Speed at 2m
        ]
    
    def fetch_weather_data(self, start_date, end_date):
        """Fetch weather data from NASA POWER API"""
        all_data = []
        
        for district, (lat, lon) in self.districts.items():
            print(f"Fetching data for {district}...")
            params = {
                'parameters': ','.join(self.parameters),
                'community': 'AG',
                'longitude': lon,
                'latitude': lat,
                'start': start_date,
                'end': end_date,
                'format': 'JSON'
            }
            
            try:
                response = requests.get(self.power_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                # Process the data
                for date, values in data['properties']['parameter'].items():
                    all_data.append({
                        'date': date,
                        'district': district,
                        'temperature': values.get('T2M', {}).get('mean', 0),
                        'humidity': values.get('RH2M', {}).get('mean', 0),
                        'rainfall': values.get('PRECTOTCORR', {}).get('mean', 0),
                        'wind_speed': values.get('WS2M', {}).get('mean', 0)
                    })
                
                # Add a small delay between requests to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error fetching data for {district}: {str(e)}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Save to CSV
        df.to_csv(os.path.join(self.data_dir, 'weather_data.csv'), index=False)
        return df
    
    def generate_crop_data(self, weather_df):
        """Generate synthetic crop yield data based on weather conditions"""
        # This is a simplified model - in reality, you should use actual crop yield data
        # from Bangladesh Bureau of Statistics
        
        # Base yield for each district (in kg/ha)
        base_yields = {
            'Dhaka': 4000,
            'Chittagong': 3800,
            'Khulna': 4200,
            'Rajshahi': 4500,
            'Sylhet': 3600,
            'Barisal': 4100,
            'Rangpur': 4300,
            'Mymensingh': 4400
        }
        
        # Calculate crop yield based on weather conditions
        crop_data = []
        for _, row in weather_df.iterrows():
            base_yield = base_yields[row['district']]
            
            # Adjust yield based on weather conditions
            temp_factor = 1.0
            if 20 <= row['temperature'] <= 30:  # Optimal temperature range
                temp_factor = 1.2
            elif row['temperature'] < 15 or row['temperature'] > 35:
                temp_factor = 0.8
            
            humidity_factor = 1.0
            if 60 <= row['humidity'] <= 80:  # Optimal humidity range
                humidity_factor = 1.1
            elif row['humidity'] < 40 or row['humidity'] > 90:
                humidity_factor = 0.9
            
            rainfall_factor = 1.0
            if 5 <= row['rainfall'] <= 15:  # Optimal rainfall range (mm/day)
                rainfall_factor = 1.15
            elif row['rainfall'] > 25:
                rainfall_factor = 0.85
            
            # Calculate final yield with some random variation
            final_yield = base_yield * temp_factor * humidity_factor * rainfall_factor
            final_yield *= np.random.uniform(0.95, 1.05)  # Add some random variation
            
            crop_data.append({
                'date': row['date'],
                'district': row['district'],
                'crop_yield': final_yield
            })
        
        # Convert to DataFrame
        df = pd.DataFrame(crop_data)
        
        # Save to CSV
        df.to_csv(os.path.join(self.data_dir, 'crop_data.csv'), index=False)
        return df

def main():
    # Initialize fetcher
    fetcher = BangladeshDataFetcher()
    
    # Set date range (last 5 years)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y%m%d')
    
    # Fetch weather data
    print("Fetching weather data...")
    weather_df = fetcher.fetch_weather_data(start_date, end_date)
    print(f"Fetched {len(weather_df)} weather records")
    
    # Generate crop data
    print("Generating crop data...")
    crop_df = fetcher.generate_crop_data(weather_df)
    print(f"Generated {len(crop_df)} crop yield records")
    
    print("Data preparation complete!")
    print(f"Weather data saved to: {os.path.join('data', 'bangladesh', 'weather_data.csv')}")
    print(f"Crop data saved to: {os.path.join('data', 'bangladesh', 'crop_data.csv')}")

if __name__ == '__main__':
    main() 