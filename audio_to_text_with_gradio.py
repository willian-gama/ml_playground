import os

import gradio
import librosa
import soundfile
from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")

def convert_audio_stereo_to_mono(audio_path):
    audi_data, sample_rate = librosa.load(audio_path, sr=model.cfg.preprocessor.sample_rate, mono=True)
    temp_file = "temp.wav"
    soundfile.write(temp_file, audi_data, sample_rate)
    return temp_file

def transcribe(audio_file):
    processed_file = convert_audio_stereo_to_mono(audio_file)
    result = model.transcribe([processed_file], source_lang="en", target_lang="en")
    os.remove(processed_file)
    return result[0].text

gradio = gradio.Interface(
    fn=transcribe,
    inputs=gradio.Audio(label="Select an audio", sources=["upload"], type="filepath"),
    outputs=gradio.Textbox(label="Transcription", lines=3, max_lines=50),
    title="Audio to text with nvidia model",
    description="Transcribe audio"
)
gradio.launch()