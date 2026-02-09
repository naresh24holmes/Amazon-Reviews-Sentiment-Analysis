import os
import re
import string
import pickle
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# Ensure NLTK resources are available
nltk.download('stopwords')
nltk.download('punkt')

# -----------------------------
# Configurable artifact paths
# -----------------------------
MODEL_FILE = "sentiment_model.pkl"
VECT_FILE = "tfidf_vectorizer.pkl"
DATA_FILE = "reviews.csv"

# -----------------------------
# Data Loading
# -----------------------------
def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    # Remove neutral reviews
    df = df[df['Score'] != 3].copy()
    return df

# -----------------------------
# Text Cleaning
# -----------------------------
def build_stopwords() -> set:
    stop_words = set(stopwords.words('english'))
    # Add domain-specific noise words
    stop_words.update(['br', 'amazon', 'product', 'one', 'get', 'would'])
    return stop_words

def clean_text(text: str, stop_words: set) -> str:
    text = re.sub(r'<.*?>', '', text)              # Remove HTML tags
    text = text.lower()                            # Lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words and word.isalpha()]
    return " ".join(tokens)

# -----------------------------
# Preprocessing
# -----------------------------
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    stop_words = build_stopwords()
    df['cleaned_text'] = df['Text'].apply(lambda t: clean_text(t, stop_words))
    df['sentiment'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)
    return df[['cleaned_text', 'sentiment']].copy()

# -----------------------------
# Model Training
# -----------------------------
def train_model(X: pd.Series, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Handle imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train_vec, y_train = smote.fit_resample(X_train_vec, y_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model, vectorizer

# -----------------------------
# Save Artifacts
# -----------------------------
def save_artifacts(model, vectorizer, model_path=MODEL_FILE, vect_path=VECT_FILE):
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    with open(vect_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"Vectorizer saved to {vect_path}")

# -----------------------------
# Main Execution
# -----------------------------
def main():
    df = load_data(DATA_FILE)
    df_model = preprocess(df)

    print(f"Dataset shape: {df_model.shape}")
    print(f"Class Balance:\n{df_model['sentiment'].value_counts()}")

    model, vectorizer = train_model(df_model['cleaned_text'], df_model['sentiment'])
    save_artifacts(model, vectorizer)

if __name__ == "__main__":
    main()





