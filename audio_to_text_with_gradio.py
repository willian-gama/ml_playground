import gradio
from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")

def transcribe(audio_file):
    result = model.transcribe([audio_file], source_lang="en", target_lang="en")
    return result[0].text

gradio = gradio.Interface(
    fn=transcribe,
    inputs=gradio.Audio(label="Select an audio", sources=["upload"], type="filepath"),
    outputs=gradio.Textbox(label="Transcription", lines=3, max_lines=50),
    title="Audio to text",
    description="Transcribe English audio"
)
gradio.launch()
