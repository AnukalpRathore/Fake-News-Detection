import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
fake = pd.read_csv(r"DATASET\archive (1)\fake.csv")
true = pd.read_csv(r"DATASET\archive (1)\true.csv")
# Labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Combine title and text
data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")

X = data["content"]
y = data["label"]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    max_df=0.7,
    min_df=2,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression Model
# Logistic Regression Model
model = LogisticRegression(
    max_iter=2000,
    C=2.0,
    class_weight="balanced",
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print(f"Accuracy : {accuracy*100:.2f}%")
print("==============================")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))

print("\nModel saved successfully!")
print("fake_news_model.pkl")
print("tfidf_vectorizer.pkl")