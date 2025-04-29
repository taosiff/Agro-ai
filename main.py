from agro_ai import AgroClimateAI
from dotenv import load_dotenv
import os

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize AgroClimateAI
    agro_ai = AgroClimateAI()
    
    print("Welcome to AgroClimateAI!")
    print("Available districts: Dhaka, Chittagong, Khulna, Rajshahi, Sylhet, Barisal, Rangpur, Mymensingh")
    
    while True:
        # Get district
        district = input("\nEnter district name (or 'quit' to exit): ").strip()
        if district.lower() == 'quit':
            break
            
        # Get query
        query = input("Enter your question: ").strip()
        if not query:
            print("Please enter a valid question.")
            continue
            
        # Process query
        response = agro_ai.process_query(query, district)
        print("\nResponse:", response)

if __name__ == "__main__":
    main() 