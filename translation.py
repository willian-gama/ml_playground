# Transformers: https://huggingface.co/docs/transformers/main_classes/pipelines
from transformers import pipeline

translator = pipeline(
    task="translation",
    model="facebook/m2m100_1.2B",
    device=1
)

text_to_translate = "Hello, how are you today?"
result = translator(
    "Hello, how's it going my friend?",
    src_lang="en",
    tgt_lang="pt"
)

print("Original:", text_to_translate)
print("Translated:", result[0]['translation_text'])
