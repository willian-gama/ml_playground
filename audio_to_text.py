from nemo.collections.asr.models import ASRModel

model = ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
audio_file = "test_audio.m4a"

transcription = model.transcribe([audio_file], source_lang="en", target_lang="en")
print("Transcription:", transcription[0].text)