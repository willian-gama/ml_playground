import torch
import soundfile as sf
from nemo.collections.asr.models import EncDecMultiTaskModel

# Device selection for macOS
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load the Canary-1B-v2 model
model = EncDecMultiTaskModel.from_pretrained("nvidia/canary-1b-v2")

# Cast model to float32 for MPS compatibility
if device == "mps":
    model = model.to(torch.float32)

model = model.to(device)

def transcribe_audio(wav_path, source_lang="en", target_lang="en"):
    # Prepare decoding configuration
    decode_cfg = model.cfg.decoding
    decode_cfg.beam.beam_size = 1
    model.change_decoding_strategy(decode_cfg)

    # Perform transcription
    result = model.transcribe([wav_path], source_lang=source_lang, target_lang=target_lang)
    return result[0].text

# Path to your fixed audio file
wav_path = "test_audio.m4a"

# Ensure the audio file exists
try:
    with open(wav_path, 'rb') as f:
        pass
except FileNotFoundError:
    print(f"Error: The file '{wav_path}' does not exist.")
else:
    transcription = transcribe_audio(wav_path)
    print("Transcription:", transcription)