import joblib
import yaml
import logging
from agro_ai import AgroClimateAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_models():
    try:
        # Initialize AgroClimateAI
        logger.info("Initializing AgroClimateAI...")
        ai = AgroClimateAI()
        
        # Test districts
        districts = ['dhaka', 'chittagong', 'khulna']
        
        for district in districts:
            logger.info(f"\nTesting predictions for {district.upper()}:")
            
            # Test weather forecast
            logger.info("1. Testing weather forecast...")
            forecast = ai.get_weather_forecast(district)
            if forecast:
                logger.info(f"✓ Weather forecast working - Got {len(forecast)} forecast points")
            else:
                logger.error("✗ Weather forecast failed")
            
            # Test real-time weather
            logger.info("2. Testing real-time weather...")
            weather = ai.get_real_time_weather(district)
            if weather:
                logger.info("✓ Real-time weather working")
                logger.info(f"  Current temperature: {weather.get('temperature', 'N/A')}°C")
                logger.info(f"  Current humidity: {weather.get('humidity', 'N/A')}%")
            else:
                logger.error("✗ Real-time weather failed")
            
            # Test flood prediction
            logger.info("3. Testing flood prediction...")
            flood_risk = ai.predict_flood_risk(district)
            if flood_risk:
                logger.info("✓ Flood prediction working")
                logger.info(f"  Risk Level: {flood_risk['risk_level']}")
                logger.info(f"  Probability: {flood_risk['probability']}%")
            else:
                logger.error("✗ Flood prediction failed")
            
            # Test query processing
            logger.info("4. Testing query processing...")
            test_queries = [
                "What's the weather like?",
                "Is there a flood risk?",
                "What's the forecast for next week?"
            ]
            
            for query in test_queries:
                response = ai.process_query(query, district)
                if response:
                    logger.info(f"✓ Query processing working for: '{query}'")
                else:
                    logger.error(f"✗ Query processing failed for: '{query}'")
        
        logger.info("\nAll tests completed!")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    test_models() 