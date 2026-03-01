
import pandas as pd
import os
import re
from src.utils.preprocess import normalize_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ------------------------------
# Directories
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ------------------------------
# NEGATION-AWARE PREPROCESSING
# ------------------------------
def preprocess_negation(text: str) -> str:
    """
    Normalize text and mark negations like 'not good' → 'not_good'
    """
    text = normalize_text(text)
    # mark negations: not|never|no + word → single token
    text = re.sub(r"\b(not|never|no)\s+(\w+)", r"\1_\2", text)
    return text

# ------------------------------
# 1. LOAD TRAIN DATA WITH ENCODING FALLBACK
# ------------------------------
train_path = os.path.join(DATA_DIR, "train.csv")
try:
    df_train = pd.read_csv(train_path, encoding="utf-8")
except UnicodeDecodeError:
    print("UTF-8 failed, trying latin1...")
    df_train = pd.read_csv(train_path, encoding="latin1")

print("Loaded train.csv:", df_train.shape)

# ------------------------------
# 2. LOAD FEEDBACK DATA (IF EXISTS)
# ------------------------------
feedback_path = os.path.join(DATA_DIR, "feedback.csv")
if os.path.exists(feedback_path):
    try:
        df_feedback = pd.read_csv(feedback_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("UTF-8 failed for feedback, trying latin1...")
        df_feedback = pd.read_csv(feedback_path, encoding="latin1")

    # convert feedback → training format
    df_feedback = df_feedback.rename(columns={"user_feedback": "sentiment"})
    df_feedback = df_feedback[["text", "sentiment"]]
    print("Loaded feedback.csv:", df_feedback.shape)

    # merge train + feedback
    df = pd.concat([df_train, df_feedback], ignore_index=True)
else:
    print("No feedback.csv found")
    df = df_train.copy()

print("Combined dataset:", df.shape)

# ------------------------------
# 3. BASIC CLEANING
# ------------------------------
df = df.dropna(subset=["text", "sentiment"]).reset_index(drop=True)

# ------------------------------
# 4. NEGATION-AWARE NORMALIZATION
# ------------------------------
df["text_normalized"] = df["text"].apply(preprocess_negation)

# ------------------------------
# 5. REMOVE DUPLICATES
# keep last → feedback overrides old label
# ------------------------------
df = df.drop_duplicates(subset=["text_normalized"], keep="last")
print("After deduplication:", df.shape)

# ------------------------------
# 6. FEATURES & LABELS
# ------------------------------
X = df["text_normalized"]
y = df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print("Train size:", X_train.shape, "Test size:", X_test.shape)

# ------------------------------
# 7. TF-IDF VECTORIZE
# ------------------------------
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),  # include bigrams for negation
    min_df=1,            # keep rare phrases like "not_good"
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vectorization complete! Train shape:", X_train_vec.shape)

# ------------------------------
# 8. MODEL TRAINING
# ------------------------------
model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)
model.fit(X_train_vec, y_train)
print("Model training completed!")

# ------------------------------
# 9. EVALUATION
# ------------------------------
y_pred = model.predict(X_test_vec)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# 10. SAVE MODEL + VECTORIZER
# ------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

print("\n✅ Model retrained with feedback (negation-aware) and saved successfully!")
