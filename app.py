import streamlit as st
import pandas as pd
import joblib
import re
import string

# Load model
model = joblib.load("logistic_regression_model.pkl")

# Page setup
st.set_page_config(
    page_title="Product Review Sentiment Analysis",
    page_icon="🛍️",
    layout="centered"
)

st.title("🛍️ Product Review Sentiment Analysis")
st.write("Predict customer review sentiment as Positive, Negative, or Neutral.")

st.markdown("---")

# Input fields
product_name = st.text_input("Product Name")

product_price = st.number_input("Product Price", min_value=0, value=1000)

rate = st.selectbox("Rating", [1, 2, 3, 4, 5])

review = st.text_input("Short Review")

summary = st.text_area("Review Summary")


# Cleaning function
def clean_text(text):
    text = re.sub("[^a-zA-Z]", " ", str(text))
    text = text.lower()
    return text


# Prediction
if st.button("Predict Sentiment"):

    if product_name == "" or review == "" or summary == "":
        st.warning("Please fill all input fields.")

    else:
        text = summary + " " + review

        word_count = len(text.split())

        input_data = pd.DataFrame({
            "product_price": [product_price],
            "Rate": [rate],
            "Char_count": [len(text)],
            "word_count": [word_count],
            "sentence_count": [len(text.split("."))],
            "avg_word_length": [
                sum(len(word) for word in text.split()) / word_count
                if word_count > 0 else 0
            ],
            "vocabulary_size": [len(set(text.split()))],
            "stopword_count": [
                sum(1 for word in text.lower().split()
                    if word in ["a", "an", "the", "is", "are", "to", "of", "in", "and", "or"])
            ],
            "punctuation_count": [
                sum(1 for ch in text if ch in string.punctuation)
            ],
            "clean_product_name": [clean_text(product_name)],
            "clean_text": [clean_text(text)]
        })

        prediction = model.predict(input_data)[0]

        st.markdown("---")
        st.subheader("Prediction Result")

        if prediction == "positive":
            st.success("😊 Positive Review")
        elif prediction == "negative":
            st.error("😞 Negative Review")
        else:
            st.info("😐 Neutral Review")

        st.write("Predicted Sentiment:", prediction.upper())


st.markdown("---")
st.caption("Product Review Sentiment Analysis using Machine Learning")