import sys
from transformers import AutoModelForSequenceClassification

models = [
    "jy46604790/Fake-News-Bert-Detect",
    "hamzab/roberta-fake-news-classification",
    "Rishabh710/fake-news-detection",
    "mrm8488/bert-tiny-finetuned-fake-news-detection"
]

for m in models:
    print(f"\nTrying {m}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(m)
        print(f"SUCCESS: {m} loaded fine!")
        print(f"Labels: {model.config.id2label}")
    except Exception as e:
        print(f"FAILED: {m} -> {type(e).__name__}: {str(e)[:100]}")
