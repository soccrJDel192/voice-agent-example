from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify that the API key was loaded
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables")
else:
    print("OpenAI API key loaded successfully")
