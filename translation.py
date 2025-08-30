from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load model and tokenizer
model_name = "facebook/m2m100_1.2B"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Set source and target languages
tokenizer.src_lang = "en"
tgt_lang = "pt"

# Encode input
inputs = tokenizer("Hello, how are you today?", return_tensors="pt")

# Generate translation
translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))

# Decode
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print("Original:", "Hello, how are you today?")
print("Translated:", translated_text)
