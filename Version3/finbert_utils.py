# finbert_utils.py

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert").to(device)
labels = ["positive", "negative", "neutral"]

def estimate_sentiment(news: List[str]) -> Tuple[str, float]:
    if not news:
        return "neutral", 0.0

    tokens = tokenizer(news, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        logits = model(tokens["input_ids"], attention_mask=tokens["attention_mask"])["logits"]

    summed_logits = torch.sum(logits, dim=0)
    probs = torch.nn.functional.softmax(summed_logits, dim=0)

    best_idx = torch.argmax(probs).item()
    sentiment_label = labels[best_idx]
    confidence = probs[best_idx].item()

    return sentiment_label, confidence

if __name__ == "__main__":
    example_headlines = ["The markets responded positively today!", "Traders are thrilled."]
    label, conf = estimate_sentiment(example_headlines)
    print("Sentiment:", label, "Confidence:", conf)
