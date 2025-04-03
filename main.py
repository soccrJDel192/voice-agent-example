from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import time

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

def speech_to_text(audio_file_path, model="gpt-4o-transcribe"):
    """
    Convert speech from an audio file to text using OpenAI's API.
    
    Args:
        audio_file_path (str): Path to the audio file to transcribe
        model (str): The model to use for transcription (default: "gpt-4o-transcribe")
    
    Returns:
        str: The transcribed text
    """
    client = initialize_openai_client()
    
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model,
            file=audio_file
        )
    
    return transcription.text

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

def main():
    """Interactive voice conversation loop."""
    print("Welcome to the Voice Assistant!")
    print("Press 'Q' to quit at any time.")
    print("Waiting for audio input...")
    
    # Initialize conversation history
    conversation_history = []
    system_prompt = "You are a helpful and friendly assistant. Keep your responses concise and natural."
    
    try:
        while True:
            # Check if user wants to quit
            user_input = input("\nPress Enter to record (or 'Q' to quit): ").strip().upper()
            if user_input == 'Q':
                print("Goodbye!")
                break
            
            # Get audio file path from user
            audio_path = input("Enter the path to your audio file: ").strip()
            if not os.path.exists(audio_path):
                print(f"Error: File not found at {audio_path}")
                continue
            
            try:
                # Transcribe the audio
                print("Transcribing audio...")
                question = speech_to_text(audio_path)
                print(f"You said: {question}")
                
                # Add to conversation history
                conversation_history.append({"role": "user", "content": question})
                
                # Get AI response and convert to speech
                print("Getting AI response...")
                response, speech_file = ask_and_speak(
                    question=question,
                    system_prompt=system_prompt,
                    voice="coral"
                )
                
                # Add AI response to conversation history
                conversation_history.append({"role": "assistant", "content": response})
                
                print(f"\nResponse saved to: {speech_file}")
                print("You can play this file to hear the response.")
                
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                continue
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
