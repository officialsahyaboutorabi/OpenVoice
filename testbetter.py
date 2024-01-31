import os
import torch
import argparse
import pyaudio
import wave
from zipfile import ZipFile
import langid
import se_extractor
from api import BaseSpeakerTTS, ToneColorConverter
import os
import time
import speech_recognition as sr
import whisper
from litellm import completion

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Initialize the Ollama client with the API key
#client = ollama(base_url="http://localhost:11434")

# Initialize the Ollama client with the API key
#ollama_client = Client("http://127.0.0.1:11434") 

# Define the name of the log file
chat_log_filename = "chatbot_conversation_log.txt"

# Function to play audio using PyAudio
def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')

    # Create a PyAudio instance
    p = pyaudio.PyAudio()

    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load models
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

# Main processing function
def process_and_play(prompt, style, audio_file_pth):
    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'default' else en_source_style_se

    speaker_wav = audio_file_pth

    # Process text and generate audio
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)

        src_path = f'{output_dir}/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')

        save_path = f'{output_dir}/output.wav'
        # Run the tone color converter
        encode_message = "@MyShell"
        tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message=encode_message)

        print("Audio generated successfully.")
        play_audio(src_path)

    except Exception as e:
        print(f"Error during audio generation: {e}")

# New function to send a query to Ollama's API, simulate streaming response, and print each full line in yellow color
def ollama_streamed(user_input, system_message, conversation_history, bot_name):
    """
    Function to send a query to Ollama's API, simulate streaming response, and print each full line in yellow color.
    Logs the conversation to a file.
    """
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
    
    # Extract content from each message and concatenate into a single string
    prompt = "\n".join(message["content"] for message in messages)

    # Simulate streaming response (replace this with the actual way Ollama handles responses)
    response_content = completion(  # Adjust this line based on Ollama API usage
        model="huggingface/TheBloke/Mistral-7B-Instruct-v0.2-GGUF",  # Specify the Ollama model name
        prompt=prompt,
        api_base="http://127.0.0.1:7861/api/generate"
    ).get('response', '')  # Update this based on the Ollama API response structure

    # Split the content into lines
    lines = response_content.split('\n')

    full_response = ""
    with open(chat_log_filename, "a") as log_file:  # Open the log file in append mode
        for line in lines:
            print(NEON_GREEN + line + RESET_COLOR)
            full_response += line + '\n'
            log_file.write(f"{bot_name}: {line}\n")  # Log the line with the bot's name

    return full_response



def transcribe_with_whisper(audio_file_path):
    # Load the model
    model = whisper.load_model("base.en")  # You can choose different model sizes like 'tiny', 'base', 'small', 'medium', 'large'

    # Transcribe the audio
    result = model.transcribe(audio_file_path)
    return result["text"]

# Function to record audio from the microphone and save to a file
def record_audio(file_path):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    frames = []

    print("Recording...")

    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print("Recording stopped.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

# New function to handle a conversation with a user
def user_chatbot_conversation():
    conversation_history = []
    system_message = open_file("chatbot1.txt")
  

    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)  # Clean up the temporary audio file

        if user_input.lower() == "exit":  # Say 'exit' to end the conversation
            break

        print(CYAN + "You:", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Jimmy:" + RESET_COLOR)
        
        # Update this line to use the Ollama API
        chatbot_response = ollama_streamed(user_input, system_message, conversation_history, "Chatbot")
        
        conversation_history.append({"role": "assistant", "content": chatbot_response})
        
        prompt2 = chatbot_response
        style = "default"
        audio_file_pth2 = "lil.mp3"
        process_and_play(prompt2, style, audio_file_pth2)

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

user_chatbot_conversation()  # Start the conversation
