from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Verify that the API key was loaded
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment variables")
    exit(1)
else:
    print("OpenAI API key loaded successfully")

# Initialize the OpenAI client with your API key
client = OpenAI(api_key=openai_api_key)

# Create a chat completion
completion = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ]
)

# Print the response
print(completion.choices[0].message)
