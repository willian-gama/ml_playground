# Transformers: https://huggingface.co/docs/transformers/main_classes/pipelines?utm_source=chatgpt.com
from transformers import pipeline

# Models: https://huggingface.co/models?pipeline_tag=text-classification&sort=downloads&search=sentiment
model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=model
)

texts = [
    # English
    "I love this app!",
    "This app is great and horrible at the same time",
    # Portuguese
    "Eu amo esse app!",
    "Esse app é ótimo e horrível ao mesmo tempo"
]

for text in texts:
    result = classifier(text)
    label = result[0]['label']
    score = result[0]['score']
    print(f"{text} -> {label} ({score:.2f})")
