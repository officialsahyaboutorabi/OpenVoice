import os
import torch
import argparse
import pyaudio
import wave
from zipfile import ZipFile
import langid
import se_extractor
import ollama
import time
import speech_recognition as sr
import whisper

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Ollama base URL
ollama_base_url = "http://localhost:11434/"

# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Function to play audio using PyAudio
def play_audio(file_path):
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)

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
en_base_speaker_tts = ollama.BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ollama.ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

# Load speaker embeddings for English
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)

# Function to process and play audio using Ollama
def process_and_play_ollama(prompt, style, audio_file_path):
    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'default' else en_source_style_se

    speaker_wav = audio_file_path

    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)

        src_path = f'{output_dir}/tmp.wav'
        tts_model.tts(prompt, src_path, speaker=style, language='English')

        save_path = f'{output_dir}/output.wav'
        ollama.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path)

        print("Audio generated successfully.")
        play_audio(src_path)

    except Exception as e:
        print(f"Error during audio generation: {e}")

# Function to transcribe audio using Whisper
def transcribe_with_whisper(audio_file_path):
    model = whisper.load_model("base.en")
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

# Function to handle a conversation with a user using Ollama
def user_chatbot_conversation():
    conversation_history = []
    system_message = open_file("chatbot1.txt")
    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file)
        user_input = transcribe_with_whisper(audio_file)
        os.remove(audio_file)

        if user_input.lower() == "exit":
            break

        print(CYAN + "You:", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Julie:" + RESET_COLOR)

        messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
        streamed_response = ollama.chat_streamed(base_url=ollama_base_url, model="llama2", messages=messages)

        full_response = ""
        line_buffer = ""

        with open("chatbot_conversation_log.txt", "a") as log_file:
            for chunk in streamed_response:
                delta_content = chunk.get("delta", {}).get("content")

                if delta_content is not None:
                    line_buffer += delta_content

                    if '\n' in line_buffer:
                        lines = line_buffer.split('\n')
                        for line in lines[:-1]:
                            print(NEON_GREEN + line + RESET_COLOR)
                            full_response += line + '\n'
                            log_file.write(f"Chatbot: {line}\n")
                        line_buffer = lines[-1]

            if line_buffer:
                print(NEON_GREEN + line_buffer + RESET_COLOR)
                full_response += line_buffer
                log_file.write(f"Chatbot: {line_buffer}\n")

        conversation_history.append({"role": "assistant", "content": full_response})

        prompt2 = full_response
        style = "default"
        audio_file_pth2 = "YOUR .mp3 voice PATH"
        process_and_play_ollama(prompt2, style, audio_file_pth2)

        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]

user_chatbot_conversation()  # Start the conversation
