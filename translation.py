Transformers: https://huggingface.co/docs/transformers/main_classes/pipelines
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load model and tokenizer
model_name = "facebook/m2m100_1.2B"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

tokenizer.src_lang = "en"
tgt_lang = "pt"

inputs = tokenizer("Hello, how are you today?", return_tensors="pt")

translated_tokens = model.generate(
    **inputs,
    forced_bos_token_id=tokenizer.get_lang_id(tgt_lang)
)
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print("Original:", "Hello, how are you today?")
print("Translated:", translated_text)

# Old
# from transformers import pipeline
#
# translator = pipeline(
#     task="translation",
#     model="facebook/m2m100_1.2B"
# )
#
# text_to_translate = "Dinner is ready"
# result = translator(
#     text_to_translate,
#     src_lang="en",
#     tgt_lang="pt",
#     clean_up_tokenization_spaces=True
# )
#
# print("Original:", text_to_translate)
# print("Translated:", result[0]['translation_text'])