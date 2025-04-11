# 🧠 Sentiment Analysis Project

## 📌 Project Overview

This project focuses on classifying **social media-style comments** into one of the following categories:
- 👍 Positive
- 👎 Negative
- 🤨 Sarcastic
- 😐 Neutral *(derived from non-sarcastic headlines)*

The aim is to build an intelligent sentiment analysis model that can understand tone, context, and even sarcasm in user-generated text, such as tweets or headlines.

---

## 📂 Datasets Used

We integrated multiple datasets to cover a wide spectrum of sentiment labels:

### 1. **Sentiment Dataset**
- **Source:** Custom dataset with `sentiment` column.
- **Labels:** `positive`, `negative`
- **Usage:** To classify standard sentiments.

### 2. **Sarcasm Headlines Dataset**
- **Source:** [Kaggle - News Headlines for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- **Files Used:**
  - `Sarcasm_Headlines_Dataset.json`
  - `Sarcasm_Headlines_Dataset_v2.json`
- **Labels:**
  - `1 → Sarcastic`
  - `0 → Neutral` (interpreted as neutral since it’s non-sarcastic)

---

## 🧹 Data Preprocessing

### Steps:
- Loaded JSON and CSV files
- Unified all formats into 2 columns: `text`, `label`
- Converted numeric values into readable labels:
  - `1 → Sarcastic`
  - `0 → Neutral`
  - `"positive"` → Positive
  - `"negative"` → Negative
- Removed unused columns like `article_link`
- Merged all data into a single dataframe
- Shuffled and saved as `final_dataset.csv`

---

## 🧪 Tools & Libraries

- **Python 3**
- **pandas** – for data handling
- **json** – for parsing raw JSON data
- **matplotlib/seaborn** – for visualization (optional)
- **sklearn** – for training models

---

## 🧠 Model Building (Planned Next)

### Potential Techniques:
- TF-IDF + Logistic Regression / SVM
- LSTM-based RNN
- Transformers (e.g., BERT, RoBERTa via HuggingFace)

---

## 🧑‍💻 File Structure

```plaintext
Sentiment-Analysis-Project/
│
├── data/
│   ├── Sarcasm_Headlines_Dataset.json
│   ├── Sarcasm_Headlines_Dataset_v2.json
│   └── your_sentiment_dataset.csv
│
├── final_dataset.csv
├── preprocessing.py
├── train_model.py
├── model.pkl (to be created)
├── app.py (Streamlit or Flask app, optional)
└── README.md
