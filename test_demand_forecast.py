import requests
import json
import sys

def test_demand_forecast(district="Dhaka"):
    """Test the demand forecasting feature by making a request to the API"""
    try:
        # Make a POST request to the demand forecast endpoint
        response = requests.post(
            "http://127.0.0.1:5000/predict/demand",
            json={"district": district}
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print(f"\nDemand Forecast for {district}:")
            print("-" * 50)
            
            if "error" in result:
                print(f"Error: {result['error']}")
                return False
            
            # Print the prediction results
            print(f"Predicted Demand: {result.get('predicted_demand', 'N/A')}")
            
            # Print confidence interval
            ci = result.get('confidence_interval', {})
            print(f"Confidence Interval: {ci.get('lower', 'N/A')} - {ci.get('upper', 'N/A')}")
            
            # Print recommendation
            print(f"Recommendation: {result.get('recommendation', 'N/A')}")
            
            # Print contributing factors
            print("\nContributing Factors:")
            factors = result.get('factors', {})
            
            weather = factors.get('weather_impact', {})
            print(f"  Weather Impact:")
            print(f"    Temperature: {weather.get('temperature', 'N/A')}°C")
            print(f"    Rainfall: {weather.get('rainfall', 'N/A')} mm")
            
            seasonal = factors.get('seasonal_factors', {})
            print(f"  Seasonal Factors:")
            print(f"    Rainy Season: {'Yes' if seasonal.get('is_rainy_season') else 'No'}")
            print(f"    Harvest Season: {'Yes' if seasonal.get('is_harvest_season') else 'No'}")
            
            return True
        else:
            print(f"Error: Request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Error testing demand forecast: {str(e)}")
        return False

if __name__ == "__main__":
    # Get district from command line argument or use default
    district = sys.argv[1] if len(sys.argv) > 1 else "Dhaka"
    
    print(f"Testing demand forecasting for district: {district}")
    success = test_demand_forecast(district)
    
    if success:
        print("\n✅ Demand forecasting feature is working correctly!")
    else:
        print("\n❌ Demand forecasting feature test failed.") 