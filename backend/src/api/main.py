from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.predict import predict_sentiment
from src.inference.feedback import save_user_feedback
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sentiment Analysis API", version="v1")

# Allow requests from your frontend
origins = [
    "http://localhost:3000",   # for local development
    "http://your-frontend-domain.com",  # for production
]

app.add_middleware(
    CORSMiddleware,
    # allow_origins=origins,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],    # allow GET, POST, PUT, etc.
    allow_headers=["*"],    # allow all headers
)


class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    sentiment: str


@app.get("/")
def health():
    return {"status": "ok", "message": "Sentiment APIs is running"}

@app.post("/api/v1/predict")
def predict(data: TextInput):
    return predict_sentiment(data.text)


@app.post("/api/v1/test")
def test(data: TextInput):
    return "test"