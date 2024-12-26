# finbert_utils.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple

labels = ["positive", "negative", "neutral"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We keep them as None until we actually need them
_tokenizer = None
_model = None

def load_finbert_if_needed():
    """Load the FinBERT tokenizer and model only if not already loaded."""
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)

def estimate_sentiment(news: List[str]) -> Tuple[str, float]:
    """
    Aggregate sentiment across multiple news headlines, returning a label and confidence.
    """
    load_finbert_if_needed()
    if not news:
        return "neutral", 0.0

    tokens = _tokenizer(news, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = _model(**tokens).logits

    # Sum across headlines
    summed_logits = torch.sum(logits, dim=0)
    probs = torch.nn.functional.softmax(summed_logits, dim=0)

    best_idx = torch.argmax(probs).item()
    sentiment_label = labels[best_idx]
    confidence = probs[best_idx].item()

    return sentiment_label, confidence

if __name__ == "__main__":
    # Simple test
    example_headlines = ["The markets responded positively today!", "Traders are thrilled."]
    label, conf = estimate_sentiment(example_headlines)
    print("Sentiment:", label, "Confidence:", conf)
