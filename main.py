from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

def initialize_openai_client():
    """Initialize and return the OpenAI client."""
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI()

def get_speech_file_path(filename="speech.mp3"):
    """Get the path for the speech file in the same directory as the script."""
    return Path(__file__).parent / filename

def text_to_speech(text, voice="coral", instructions=None, output_file="speech.mp3"):
    """
    Convert text to speech using OpenAI's API and save it to a file.
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: "coral")
        instructions (str, optional): Additional instructions for the speech generation
        output_file (str): The name of the output file (default: "speech.mp3")
    
    Returns:
        Path: The path to the generated speech file
    """
    client = initialize_openai_client()
    speech_file_path = get_speech_file_path(output_file)
    
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions=instructions,
    ) as response:
        response.stream_to_file(speech_file_path)
    
    return speech_file_path

def chat_completion(messages, model="gpt-4o"):
    """
    Get a chat completion from OpenAI's API.
    
    Args:
        messages (list): List of message dictionaries with 'role' and 'content'
        model (str): The model to use (default: "gpt-4o")
    
    Returns:
        str: The content of the assistant's response
    """
    client = initialize_openai_client()
    
    completion = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    return completion.choices[0].message.content

def ask_and_speak(question, system_prompt="You are a helpful assistant.", voice="coral"):
    """
    Ask a question to the AI and convert its response to speech.
    
    Args:
        question (str): The question to ask
        system_prompt (str): The system prompt for the AI (default: "You are a helpful assistant.")
        voice (str): The voice to use for speech (default: "coral")
    
    Returns:
        tuple: (response_text, speech_file_path)
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # Get the AI's response
    response = chat_completion(messages)
    print(f"AI Response: {response}")
    
    # Convert the response to speech
    speech_file = text_to_speech(
        text=response,
        voice=voice,
        instructions="Speak in a natural and conversational tone."
    )
    
    return response, speech_file

def interactive_chat_and_speak(system_prompt="You are a helpful assistant.", voice="coral"):
    """
    Start an interactive chat session where each AI response is converted to speech.
    
    Args:
        system_prompt (str): The system prompt for the AI (default: "You are a helpful assistant.")
        voice (str): The voice to use for speech (default: "coral")
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    print("Starting interactive chat session. Type 'quit' to exit.")
    print("System prompt:", system_prompt)
    print("-" * 50)
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check if user wants to quit
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Ending chat session. Goodbye!")
            break
        
        # Add user message to conversation history
        messages.append({"role": "user", "content": user_input})
        
        try:
            # Get AI response
            response = chat_completion(messages)
            print(f"\nAI: {response}")
            
            # Convert response to speech
            speech_file = text_to_speech(
                text=response,
                voice=voice,
                instructions="Speak in a natural and conversational tone."
            )
            print(f"Speech saved to: {speech_file}")
            
            # Add AI response to conversation history
            messages.append({"role": "assistant", "content": response})
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

def main():
    """Example usage of asking a question and getting a spoken response."""
    try:
        # Example question
        question = "What is the meaning of life?"
        
        # Get response and convert to speech
        response, speech_file = ask_and_speak(
            question=question,
            system_prompt="You are a wise philosopher who gives thoughtful but concise answers.",
            voice="coral"
        )
        
        print(f"\nQuestion: {question}")
        print(f"Response: {response}")
        print(f"Speech file generated at: {speech_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
