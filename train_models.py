from model_trainer import AgroClimateModels
import os
import joblib
import pandas as pd
from datetime import datetime

def main():
    # Initialize models
    models = AgroClimateModels()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    models.load_and_preprocess_data()
    
    # Train models
    print("Training models...")
    models.train_models()
    
    # Test predictions for all districts
    districts = [
        'Dhaka', 'Chittagong', 'Khulna', 'Rajshahi',
        'Sylhet', 'Barisal', 'Rangpur', 'Mymensingh'
    ]
    
    print("\nMaking predictions for all districts...")
    predictions = []
    for district in districts:
        result = models.get_predictions(district)
        predictions.append({
            'district': district,
            'crop_yield': result['crop_yield'],
            'disaster_risk': result['disaster_risk'],
            'temperature': result['weather']['temperature'],
            'humidity': result['weather']['humidity'],
            'wind_speed': result['weather']['wind_speed'],
            'rainfall': result['weather']['rainfall']
        })
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame(predictions)
    os.makedirs('data/bangladesh', exist_ok=True)
    predictions_df.to_csv('data/bangladesh/predictions.csv', index=False)
    print("\nPredictions saved to data/bangladesh/predictions.csv")
    
    # Print summary
    print("\nPrediction Summary:")
    print(predictions_df.to_string())
    
    # Save model performance metrics
    metrics = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'crop_yield_mean': predictions_df['crop_yield'].mean(),
        'crop_yield_std': predictions_df['crop_yield'].std(),
        'disaster_risk_mean': predictions_df['disaster_risk'].mean(),
        'disaster_risk_std': predictions_df['disaster_risk'].std()
    }
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv('data/bangladesh/model_metrics.csv', index=False)
    print("\nModel metrics saved to data/bangladesh/model_metrics.csv")

if __name__ == '__main__':
    main() 