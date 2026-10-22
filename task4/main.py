import os
import re
import time
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from threading import Lock
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, Trainer,
    TrainingArguments, DataCollatorWithPadding, pipeline
)
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
import joblib

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "distilbert_sentiment_model"
TFIDF_PATH = "tfidf_sentiment_pipeline.joblib"
LABELS = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_TO_ID = {v: k for k, v in LABELS.items()}

def clean_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()

# Self-contained labelled dataset (360 texts) for this assessment.
positive = [
    "I love the excellent service", "This product works perfectly", "Very happy with my purchase",
    "The support team was helpful", "Fast delivery and great quality", "Amazing experience overall",
    "The app is easy to use", "Friendly staff solved my problem", "Wonderful value for money",
    "Everything was smooth and reliable", "I am satisfied with the result", "Great product and service"
]
negative = [
    "I hate this terrible service", "The product is broken", "Very unhappy with my purchase",
    "The support team was rude", "Slow delivery and poor quality", "Awful experience overall",
    "The app is difficult to use", "Nobody solved my problem", "Waste of money",
    "Everything was confusing and unreliable", "I am disappointed with the result", "Bad product and service"
]
neutral = [
    "The product arrived today", "The service was acceptable", "I have a question about my order",
    "The package contains two items", "The app has a new update", "My order is being processed",
    "The delivery came this morning", "I contacted support about my account", "The product has basic features",
    "The website shows my order status", "I received the standard package", "The service is available now"
]
suffixes = ["today", "for my account", "after the update", "during checkout", "from the website",
            "this week", "for my order", "in the mobile app", "as expected", "for the first time"]
texts, labels = [], []
for label, templates in [(2, positive), (0, negative), (1, neutral)]:
    for i in range(120):
        texts.append(clean_text(templates[i % len(templates)] + " " + suffixes[i % len(suffixes)]))
        labels.append(label)
X_train, X_test, y_train, y_test = train_test_split(
    np.array(texts), np.array(labels), test_size=0.2, random_state=SEED, stratify=labels
)

# Always create a baseline artifact so API fallback is available.
if not os.path.exists(TFIDF_PATH):
    baseline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED))
    ])
    baseline.fit(X_train, y_train)
    joblib.dump(baseline, TFIDF_PATH)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
for sentence in [
    "I love the service and the product is excellent.",
    "The delivery was delayed and I am disappointed."
]:
    encoded = tokenizer(sentence)
    print("\nSentence:", sentence)
    print("Tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"]))
    print("Input IDs:", encoded["input_ids"])
    print("Attention mask:", encoded["attention_mask"])

print("\nSentiment pipeline:")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if torch.cuda.is_available() else -1
)
for sentence in [
    "I absolutely love this product.",
    "The service was terrible and slow.",
    "The package arrived this morning."
]:
    print(sentence, "->", sentiment_pipe(sentence))

print("\nNER pipeline:")
ner_pipe = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1)
for sentence in [
    "Sundar Pichai works at Google in California.",
    "Mary visited London last Monday.",
    "OpenAI was founded in San Francisco."
]:
    print(sentence, "->", ner_pipe(sentence))

class TextDataset(Dataset):
    def __init__(self, sentences, target_labels, tokenizer):
        self.encodings = tokenizer(list(sentences), truncation=True, max_length=128)
        self.labels = list(map(int, target_labels))
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item

train_dataset = TextDataset(X_train, y_train, tokenizer)
test_dataset = TextDataset(X_test, y_test, tokenizer)
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3, id2label=LABELS, label2id=LABEL_TO_ID
)
def compute_metrics(eval_pred):
    logits, targets = eval_pred
    pred = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(targets, pred),
            "macro_f1": f1_score(targets, pred, average="macro")}

args = TrainingArguments(
    output_dir="distilbert_training_output",
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="no",
    report_to="none",
    seed=SEED,
    fp16=torch.cuda.is_available()
)
trainer = Trainer(model=model, args=args, train_dataset=train_dataset, eval_dataset=test_dataset,
                  data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                  compute_metrics=compute_metrics)
start = time.perf_counter()
trainer.train()
train_seconds = time.perf_counter() - start
metrics = trainer.evaluate()
print("Fine-tuned DistilBERT metrics:", metrics)
trainer.save_model(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

baseline = joblib.load(TFIDF_PATH)
baseline_pred = baseline.predict(X_test)
print("\nModel comparison")
print(f"TF-IDF: accuracy={accuracy_score(y_test, baseline_pred):.4f}, "
      f"macro F1={f1_score(y_test, baseline_pred, average='macro'):.4f}")
print(f"DistilBERT: accuracy={metrics.get('eval_accuracy', 0):.4f}, "
      f"macro F1={metrics.get('eval_macro_f1', 0):.4f}, training seconds={train_seconds:.2f}")

api_lock = Lock()
resources = {"model": None, "tokenizer": None, "tfidf": None, "backend": None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.path.isdir(MODEL_DIR):
        resources["tokenizer"] = AutoTokenizer.from_pretrained(MODEL_DIR)
        resources["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE).eval()
        resources["backend"] = "distilbert"
    else:
        resources["tfidf"] = joblib.load(TFIDF_PATH)
        resources["backend"] = "tfidf"
    yield

app = FastAPI(title="Week 3 Sentiment Classifier", lifespan=lifespan)

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)

class PredictResponse(BaseModel):
    label: str
    confidence: float

@app.get("/health")
def health():
    return {"status": "ok", "backend": resources["backend"] or "not_loaded"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    with api_lock:
        if resources["backend"] == "distilbert" and resources["model"] is not None:
            encoded = resources["tokenizer"](text, return_tensors="pt", truncation=True, max_length=128)
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            with torch.no_grad():
                probabilities = torch.softmax(resources["model"](**encoded).logits, dim=-1)[0].cpu().numpy()
            index = int(np.argmax(probabilities))
            label_map = resources["model"].config.id2label
            label = label_map.get(index, label_map.get(str(index), LABELS[index]))
            if str(label).startswith("LABEL_"):
                label = LABELS[index]
            return {"label": str(label), "confidence": float(probabilities[index])}
        if resources["tfidf"] is None:
            resources["tfidf"] = joblib.load(TFIDF_PATH)
        probabilities = resources["tfidf"].predict_proba([clean_text(text)])[0]
        index = int(np.argmax(probabilities))
        class_id = int(resources["tfidf"].classes_[index])
        return {"label": LABELS[class_id], "confidence": float(probabilities[index])}

if __name__ == "__main__":
    print("\nFastAPI app is ready.")
    print("Run separately with: uvicorn main:app --host 127.0.0.1 --port 8000")
    print("POST /predict JSON example: {\"text\": \"I was charged twice\"}")
    print("BERT vs GPT:")
    print("1. BERT is an encoder model that reads context from both directions.")
    print("2. GPT is a decoder model that predicts the next token.")
    print("3. BERT is commonly used for classification and language understanding.")
    print("4. GPT is commonly used for generation and conversation.")
    print("5. For feedback classification, fine-tuned DistilBERT is suitable; TF-IDF is simpler and faster.")
