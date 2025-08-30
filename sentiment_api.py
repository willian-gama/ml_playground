# https://huggingface.co/docs/huggingface_hub/en/package_reference/inference_client
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

hf_token = os.environ.get("HF_TOKEN")
client = InferenceClient(
    provider="hf-inference", # hf-inference is already set by default
    token=hf_token
)

labels_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

examples = [
    "I like this app!", # Positive
    "This app has pros and cons.", # Neutral
    "I dislike this app!", # Negative
]

for text in examples:
    result = client.text_classification(
        model="cardiffnlp/twitter-roberta-base-sentiment",
        text=text
    )[0]
    raw_label = result['label']
    label = labels_map.get(raw_label)
    score = result['score'] * 100
    print(f"{text} -> {label} ({score:.2f})")

