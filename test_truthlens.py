import os
import sys

# Load .env
from dotenv import load_dotenv
load_dotenv()

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from truthlens.classifier import bert_classify
from truthlens.baseline import train_xgb_pipeline, xgb_pipeline_predict
from truthlens.aggregator import compute_final_verdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

print("Loading models...")
_PRIMARY_MODEL = "hamzab/roberta-fake-news-classification"
try:
    _bert_tokenizer = AutoTokenizer.from_pretrained(_PRIMARY_MODEL)
    _bert_model = AutoModelForSequenceClassification.from_pretrained(_PRIMARY_MODEL)
except Exception as e:
    print(f"Failed to load BERT: {e}")
    sys.exit(1)

_bert_model.eval()

print("Training XGBoost...")
_xgb_pipeline = train_xgb_pipeline()

test_cases = [
    "The capital of France is Paris. It is a beautiful city with the Eiffel Tower.",
    "Apple announced a new iPhone today with a faster processor and better camera.",
    "Breaking: Joe Biden has resigned from the presidency and appointed a golden retriever as his successor.",
    "COVID-19 vaccines contain 5G tracking microchips funded by Bill Gates."
]

for idx, text in enumerate(test_cases):
    print(f"\n--- Test Case {idx + 1} ---")
    print(f"Text: {text}")
    
    bert_result = bert_classify(text, _bert_tokenizer, _bert_model)
    print(f"BERT result: {bert_result}")
    
    xgb_result = xgb_pipeline_predict(text, _xgb_pipeline)
    print(f"XGBoost result: {xgb_result}")
    
    fact_results = []
    web_results = []
    
    final_verdict = compute_final_verdict(bert_result, xgb_result, fact_results, web_results)
    print(f"Final Verdict: {final_verdict}")
