# import os
# import pandas as pd
# from src.utils.preprocess import normalize_text
# from pydantic import BaseModel

# TRAIN_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "train.csv")

# class FeedbackInput(BaseModel):
#     text: str
#     sentiment: str

# def save_user_feedback(feedback: FeedbackInput):
#     # Normalize text
#     text_norm = normalize_text(feedback.text)

#     # Load CSV
#     df = pd.read_csv(TRAIN_CSV, encoding='utf-8')

#     # Check if this text + sentiment already exists
#     if ((df['text_normalized'] == text_norm) & (df['sentiment'] == feedback.sentiment)).any():
#         return False  # Already exists, skip saving

#     # Create new row
#     new_row = {
#         "text": feedback.text,
#         "text_normalized": text_norm,
#         "sentiment": feedback.sentiment
#     }

#     # Append and save
#     df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
#     df.to_csv(TRAIN_CSV, index=False)

#     return True



import os
import pandas as pd
from src.utils.preprocess import normalize_text

FEEDBACK_CSV = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "feedback.csv")

def save_user_feedback(text, predicted, user_feedback):
    # Ensure folder exists
    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    
    # Ensure CSV exists
    if not os.path.exists(FEEDBACK_CSV):
        pd.DataFrame(columns=["text", "predicted", "user_feedback"]).to_csv(FEEDBACK_CSV, index=False)

    # Normalize text same as training
    text_norm = normalize_text(text)

    # Load CSV
    df = pd.read_csv(FEEDBACK_CSV)

    # Avoid duplicates
    if not ((df["text"] == text) & (df["user_feedback"] == user_feedback)).any():
        new_row = pd.DataFrame([{
            "text": text,
            "predicted": predicted,
            "user_feedback": user_feedback
        }])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(FEEDBACK_CSV, index=False)
        print(f"Feedback saved: '{text}' -> {user_feedback}")
    else:
        print("Duplicate feedback ignored.")