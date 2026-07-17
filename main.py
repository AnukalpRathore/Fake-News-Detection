import pandas as pd
import re
import string
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
fake = pd.read_csv("DATASET/archive (1)/fake.csv")
true = pd.read_csv("DATASET/archive (1)/true.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Combine title + text
data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")

# Keep required columns
data = data[["content", "label"]]

# Shuffle
data = data.sample(frac=1, random_state=42)

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

data["content"] = data["content"].apply(clean_text)

X = data["content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train Model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))
print("\nClassification Report:")
print(classification_report(y_test, pred))

# Save model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel Saved Successfully!")

# Test
while True:
    news = input("\nEnter News (type exit to quit): ")

    if news.lower() == "exit":
        break

    news = clean_text(news)
    news_vector = vectorizer.transform([news])
    result = model.predict(news_vector)

    if result[0] == 1:
        print("🟢 Real News")
    else:
        print("🔴 Fake News")