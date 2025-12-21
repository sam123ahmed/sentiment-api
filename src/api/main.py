from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import predict_sentiment

app = FastAPI(title="Sentiment Analysis API", version="v1")

class TextInput(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "ok", "message": "Sentiment API is running"}

@app.post("/api/v1/predict")
def predict(data: TextInput):
    return predict_sentiment(data.text)

@app.post("/api/v1/testing")
def predict():
    return print("testing")