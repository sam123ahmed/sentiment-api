import joblib
import os
from src.utils.preprocess import normalize_text

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

def predict_sentiment(text: str) -> dict:
    clean_text = normalize_text(text)
    vec = vectorizer.transform([clean_text])
    prediction = model.predict(vec)[0]
    probability = model.predict_proba(vec).max()

    return {
        "text": text,
        "sentiment": prediction,
        "confidence": round(float(probability), 4)
    }