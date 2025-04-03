from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import time
import pyaudio
import wave
import keyboard

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

def record_audio(filename="input.wav", duration=5):
    """
    Record audio from the microphone and save it to a file.
    
    Args:
        filename (str): Name of the output file
        duration (int): Recording duration in seconds
    
    Returns:
        str: Path to the recorded audio file
    """
    # Audio recording parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    p = pyaudio.PyAudio()
    
    # Open stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print(f"Recording for {duration} seconds...")
    frames = []
    
    # Record for the specified duration
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Recording finished")
    
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Save the recorded data
    output_path = get_speech_file_path(filename)
    wf = wave.open(str(output_path), 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return str(output_path)

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
    print("Press 'R' to start recording (5 seconds)")
    
    # Initialize conversation history
    conversation_history = []
    system_prompt = "You are a helpful and friendly assistant. Keep your responses concise and natural."
    
    try:
        while True:
            # Check if user wants to quit
            user_input = input("\nPress 'R' to record or 'Q' to quit: ").strip().upper()
            if user_input == 'Q':
                print("Goodbye!")
                break
            elif user_input != 'R':
                continue
            
            try:
                # Record audio
                print("\nStarting recording...")
                audio_path = record_audio()
                
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
