import pandas as pd
import os
import re
from src.utils.preprocess import normalize_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import classification_report, accuracy_score



# -------------------------------------------------------------------------
# 1. LOAD CSV
# -------------------------------------------------------------------------
csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'train.csv')

try:
    df = pd.read_csv(csv_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(csv_path, encoding='latin1')

print("Loaded dataset:", df.shape)


# -------------------------------------------------------------------------
# 2. BASIC CLEANING (remove missing)
# -------------------------------------------------------------------------
df = df.dropna(subset=["text", "sentiment"])
df = df.reset_index(drop=True)


# -------------------------------------------------------------------------
# 3. REMOVE EXACT DUPLICATES
# -------------------------------------------------------------------------
df = df.drop_duplicates(subset=["text", "sentiment"])
print("After dropping exact duplicates:", df.shape)


# -------------------------------------------------------------------------
# 4. SIMPLE CLEANING FOR NEAR-DUPLICATES
# -------------------------------------------------------------------------
df["text_clean"] = df["text"].astype(str).str.lower().str.strip()
df = df.drop_duplicates(subset=["text_clean", "sentiment"])
print("After removing simple near-duplicates:", df.shape)


# -------------------------------------------------------------------------
# 5. ADVANCED NORMALIZATION
# -------------------------------------------------------------------------
# def normalize_text(text):
#     text = str(text).lower().strip()
#     text = text.replace("`", "'")               # normalize quotes
#     text = re.sub(r"[^a-z0-9\s']", "", text)    # remove punctuation except '
#     text = re.sub(r"\s+", " ", text)             # collapse spaces
#     return text

df["text_normalized"] = df["text"].apply(normalize_text)

# Drop duplicate normalized text
df = df.drop_duplicates(subset=["text_normalized", "sentiment"])
print("After advanced normalization:", df.shape)


# -------------------------------------------------------------------------
# 6. SELECT FEATURES AND LABELS
# -------------------------------------------------------------------------
X = df["text_normalized"]
y = df["sentiment"]

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 3),
    min_df=2,
    stop_words='english'
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Vectorization complete!")
print("Train vectorized shape:", X_train_vec.shape)


model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train_vec, y_train)

print("Model training completed!")

y_pred = model.predict(X_test_vec)

print("\nAccuracy: ", accuracy_score(y_test, y_pred))
print("\nclassification report: ", classification_report(y_test, y_pred))
 
# -------------------------------------------------------------------------
# 10. SAVE MODEL + TF-IDF
# -------------------------------------------------------------------------
model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(model, os.path.join(model_dir, "sentiment_model.pkl"))
joblib.dump(vectorizer, os.path.join(model_dir, "tfidf_vectorizer.pkl"))

print("Model + Vectorizer saved successfully!")