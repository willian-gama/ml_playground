# Transformers: https://huggingface.co/docs/transformers/main_classes/pipelines?utm_source=chatgpt.com
from transformers import pipeline
import gradio as gradio

# Models: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment
model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=model
)

# Gradio function
def classify_text(text):
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    return f"{label} ({score:.2f})"

# Example texts
examples = [
    "I like this app!", # Positive
    "I dislike this app!", # Neutral
    "I hate this app!" # Negative
]

# Gradio interface - find all components: https://www.gradio.app/docs/gradio/introduction#events
gradio_interface = gradio.Interface(
    fn=classify_text,
    inputs=gradio.Textbox(label="input",lines=3, placeholder="Type a sentence here..."),
    outputs=gradio.Textbox(label="output"),
    examples=examples,
    title="Multilingual Sentiment Analyzer",
    description="Enter text and see the sentiment."
)

# Launch
gradio_interface.launch()