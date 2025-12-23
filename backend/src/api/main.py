from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import predict_sentiment
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sentiment Analysis API", version="v1")

# Allow requests from your frontend
origins = [
    "http://localhost:3000",   # for local development
    "http://your-frontend-domain.com",  # for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # list of allowed origins
    allow_credentials=True,
    allow_methods=["*"],    # allow GET, POST, PUT, etc.
    allow_headers=["*"],    # allow all headers
)


class TextInput(BaseModel):
    text: str


@app.get("/")
def health():
    return {"status": "ok", "message": "Sentiment API is running"}

@app.post("/api/v1/predict")
def predict(data: TextInput):
    return predict_sentiment(data.text)