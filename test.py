# test.py

import joblib
import re

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower().strip()
    return text

# Load the trained model once
model = joblib.load("sentiment_model.pkl")

print("🧠 Sentiment & Sarcasm Classifier")
print("Type 'exit' to quit.\n")

while True:
    # Take input from user
    user_input = input("💬 Enter a social media comment:\n> ")

    if user_input.strip().lower() in ['exit', 'quit']:
        print("👋 Exiting...")
        break

    cleaned_input = clean_text(user_input)
    predicted_label = model.predict([cleaned_input])[0]

    print(f"📌 Predicted Sentiment: {predicted_label}\n")
