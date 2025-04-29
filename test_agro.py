from agro_ai import AgroClimateAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    # Initialize the AgroClimateAI
    agro_ai = AgroClimateAI()
    
    # Test with Dhaka
    district = "Dhaka"
    
    # Test queries
    queries = [
        "What's the current temperature?",
        "Will it rain today?",
        "What's the weather forecast for tomorrow?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        response = agro_ai.process_query(query, district)
        print(f"Response: {response}")

if __name__ == "__main__":
    main() 