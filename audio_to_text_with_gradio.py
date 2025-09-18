import gradio
import soundfile as sf
import librosa
from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")

def convert_audio_stereo_to_mono(audio_path):
    y, sr = librosa.load(audio_path, sr=model.cfg.preprocessor.sample_rate, mono=True)
    temp_file = "temp.wav"
    sf.write(temp_file, y, sr)
    return temp_file

def transcribe(audio_file):
    processed_file = convert_audio_stereo_to_mono(audio_file)
    result = model.transcribe([processed_file], source_lang="en", target_lang="en")
    return result[0].text

gradio = gradio.Interface(
    fn=transcribe,
    inputs=gradio.Audio(label="Select an audio", sources=["upload"], type="filepath"),
    outputs=gradio.Textbox(label="Transcription", lines=3, max_lines=50),
    title="Audio to text",
    description="Transcribe audio"
)
gradio.launch()
