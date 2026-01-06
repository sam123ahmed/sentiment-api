import os
import pandas as pd
from src.utils.preprocess import normalize_text

TRAIN_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train.csv")

def save_user_feedback(feedback: FeedbackInput):
    # Normalize text
    text_norm = normalize_text(feedback.text)

    # Load CSV
    df = pd.read_csv(TRAIN_CSV, encoding='utf-8')

    # Check if this text + sentiment already exists
    if ((df['text_normalized'] == text_norm) & (df['sentiment'] == feedback.sentiment)).any():
        return False  # Already exists, skip saving

    # Create new row
    new_row = {
        "text": feedback.text,
        "text_normalized": text_norm,
        "sentiment": feedback.sentiment
    }

    # Append and save
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(TRAIN_CSV, index=False)

    return True
