from agro_ai import AgroClimateAI
import sys

def main():
    print("Welcome to AgroClimate AI Chat Interface!")
    print("----------------------------------------")
    print("This system helps you with agricultural and climate information for Bangladesh.")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("First, please choose a district from: dhaka, chittagong, khulna, rajshahi, sylhet, barisal, rangpur, mymensingh")
    
    try:
        # Initialize the AI system
        ai = AgroClimateAI()
        
        # Get the district
        district = input("\nEnter district name: ").strip().lower()
        while district not in ['dhaka', 'chittagong', 'khulna', 'rajshahi', 'sylhet', 'barisal', 'rangpur', 'mymensingh']:
            print("Please enter a valid district name from the list above.")
            district = input("Enter district name: ").strip().lower()
        
        print(f"\nGreat! You selected {district.title()}. How can I help you today?")
        print("You can ask about weather, crop recommendations, or any agricultural concerns.")
        
        # Main chat loop
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Check for exit command
            if user_input.lower() in ['quit', 'exit']:
                print("\nThank you for using AgroClimate AI. Goodbye!")
                break
            
            # Get AI response
            try:
                response = ai.process_query(user_input, district)
                print("\nAI:", response)
            except Exception as e:
                print("\nSorry, I encountered an error. Please try again.")
                print(f"Error details: {str(e)}")
    
    except Exception as e:
        print("\nError initializing the AI system.")
        print(f"Error details: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 