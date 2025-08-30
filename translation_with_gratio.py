import gradio as gradio
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load model and tokenizer
model_name = "facebook/m2m100_1.2B"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# Translation function
def translate_text(text, source_lang, target_lang):
    tokenizer.src_lang = source_lang
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id(target_lang)
    )
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Examples
examples = [
    ["Hi, How are you?", "en", "pt"],
    ["Let's have a call", "en", "pt"],
    ["The dinner is ready", "en", "pt"]
]

# Gradio interface
gradio_interface = gradio.Interface(
    fn=translate_text,
    inputs=[
        gradio.Textbox(lines=2, label="Text to translate", placeholder="Enter text here..."),
        gradio.Textbox(value="en", label="Source language code (en)"),
        gradio.Textbox(value="pt", label="Target language code (pt)")
    ],
    outputs=gradio.Textbox(label="Translated text"),
    title="Multilingual Translation",
    description="Translate text from any language using M2M100",
    examples=examples
)

gradio_interface.launch()