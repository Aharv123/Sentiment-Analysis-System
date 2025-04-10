# train.py

import pandas as pd
import json
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

# ---------- Load and Prepare Datasets ----------
def load_sarcasm_dataset(path):
    with open(path, 'r') as f:
        return pd.DataFrame([json.loads(line) for line in f])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower().strip()
    return text

# ---------- Load Sentiment Dataset ----------
df_sent = pd.read_csv("train.csv", encoding='ISO-8859-1')[['text', 'sentiment']]
df_sent.rename(columns={'sentiment': 'label'}, inplace=True)

# Map sentiment to simplified categories
df_sent['label'] = df_sent['label'].map({
    'positive': 'good',
    'negative': 'bad',
    'neutral': 'neutral'
})

# ---------- Load Sarcasm Datasets ----------
df_sarc1 = load_sarcasm_dataset("Sarcasm_Headlines_Dataset.json")
df_sarc2 = load_sarcasm_dataset("Sarcasm_Headlines_Dataset_v2.json")
df_sarc = pd.concat([df_sarc1, df_sarc2], ignore_index=True)[['headline', 'is_sarcastic']]
df_sarc.rename(columns={'headline': 'text', 'is_sarcastic': 'label'}, inplace=True)
df_sarc['label'] = df_sarc['label'].map({1: 'sarcastic', 0: 'neutral'})  # Treat not sarcastic as neutral

# ---------- Downsample Sarcasm Dataset ----------
# Keep sarcasm data in proportion to sentiment data
sent_len = len(df_sent)
df_sarc = df_sarc.sample(n=sent_len, random_state=42)  # balance it

# ---------- Combine & Clean ----------
df_combined = pd.concat([df_sent, df_sarc], ignore_index=True)
df_combined['text'] = df_combined['text'].apply(clean_text)
df_combined.dropna(subset=['text', 'label'], inplace=True)
df_combined = shuffle(df_combined, random_state=42)

# ---------- Train/Test Split ----------
X = df_combined['text']
y = df_combined['label']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Model Pipeline ----------
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=300, class_weight='balanced'))
])

# ---------- Train ----------
model.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = model.predict(X_val)
print("\n📊 Classification Report:\n")
print(classification_report(y_val, y_pred))

# ---------- Save ----------
joblib.dump(model, "sentiment_model.pkl")
print("✅ Model trained and saved as sentiment_model.pkl")
