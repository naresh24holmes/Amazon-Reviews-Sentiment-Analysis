import streamlit as st
import pickle

# Load artifacts
MODEL_FILE = "sentiment_model.pkl"
VECT_FILE = "sentiment_vectorizer.pkl"

with open(MODEL_FILE, "rb") as f:
    model = pickle.load(f)

with open(VECT_FILE, "rb") as f:
    vectorizer = pickle.load(f)

# Streamlit UI
st.title("Sentiment Analysis App")
st.write("Enter a product review and get sentiment prediction (Positive/Negative).")

# Text input
user_input = st.text_area("Review Text", "")

if st.button("Predict"):
    if user_input.strip():
        # Transform input
        X_vec = vectorizer.transform([user_input])
        prediction = model.predict(X_vec)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😡"
        st.success(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter some text before predicting.")
