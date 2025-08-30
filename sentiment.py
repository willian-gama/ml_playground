# Transformers: https://huggingface.co/docs/transformers/main_classes/pipelines
from transformers import pipeline

# Models: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment
model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
output = pipeline(
    task="text-classification",
    model=model,
    tokenizer=model
)

examples = [
    "I like this app!",  # Positive
    "I dislike this app!",  # Neutral
    "Worst app I have ever used."  # Negative
]

for text in examples:
    result = output(text)[0]
    label = result['label']
    score = result['score']
    print(f"{text} -> {label} ({score:.2f})")
