import re
import time
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SEED = 42
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
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

suffixes = ["today", "for my account", "after the update", "during checkout", "from the website",
            "this week", "for my order", "in the mobile app", "as expected", "for the first time"]
texts, labels = [], []
for label, templates in [(2, positive), (0, negative), (1, neutral)]:
    for i in range(120):
        texts.append(clean_text(templates[i % len(templates)] + " " + suffixes[i % len(suffixes)]))
        labels.append(label)
texts, labels = np.array(texts), np.array(labels)
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("classifier", LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED))
])
start = time.perf_counter()
model.fit(X_train, y_train)
training_seconds = time.perf_counter() - start
start = time.perf_counter()
predictions = model.predict(X_test)
inference_seconds = (time.perf_counter() - start) / len(X_test)
print("Dataset size:", len(texts))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Macro F1:", round(f1_score(y_test, predictions, average="macro"), 4))
print("Confusion matrix (0=negative, 1=neutral, 2=positive):\n", confusion_matrix(y_test, predictions, labels=[0, 1, 2]))
print("Training seconds:", round(training_seconds, 3))
print("Inference seconds per text:", round(inference_seconds, 6))

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]
terms = np.array(vectorizer.get_feature_names_out())
names = {0: "negative", 1: "neutral", 2: "positive"}
for row, class_id in enumerate(classifier.classes_):
    top = np.argsort(classifier.coef_[row])[-10:][::-1]
    print(f"Top words for {names[int(class_id)]}:", terms[top].tolist())

joblib.dump(model, "tfidf_sentiment_pipeline.joblib")
np.savez("test_data.npz", X_test=X_test, y_test=y_test)
print("Saved tfidf_sentiment_pipeline.joblib and test_data.npz")

try:
    import gensim.downloader as api
    print("Loading GloVe vectors (internet needed on first run)...")
    glove = api.load("glove-wiki-gigaword-50")
    for word in ["computer", "happy"]:
        print(f"Nearest words to {word}:", glove.most_similar(word, topn=5))
    print("king - man + woman:", glove.most_similar(positive=["king", "woman"], negative=["man"], topn=1))
    def average_vector(sentence):
        vectors = [glove[word] for word in clean_text(sentence).split() if word in glove]
        return np.mean(vectors, axis=0) if vectors else np.zeros(50)
    X_emb_train = np.vstack([average_vector(t) for t in X_train])
    X_emb_test = np.vstack([average_vector(t) for t in X_test])
    emb_classifier = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=SEED)
    emb_classifier.fit(X_emb_train, y_train)
    emb_pred = emb_classifier.predict(X_emb_test)
    print("Average GloVe accuracy:", round(accuracy_score(y_test, emb_pred), 4))
    print("Average GloVe macro F1:", round(f1_score(y_test, emb_pred, average="macro"), 4))
except Exception as exc:
    print("GloVe section skipped:", repr(exc))
    print("Install gensim and enable internet access to run embeddings.")
