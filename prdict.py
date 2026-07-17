import pickle

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

print("=" * 40)
print("       FAKE NEWS DETECTION")
print("=" * 40)
while True:
    news = input("\nEnter News (or type 'exit' to quit): ")

    if news.lower() == "exit":
        break

    news_vector = vectorizer.transform([news])

    prediction = model.predict(news_vector)[0]

    # Confidence using decision_function
    probs = model.predict_proba(news_vector)[0]
    confidence = max(probs) * 100

    print("\n========== RESULT ==========")

    if prediction == 1:
        print("Prediction : REAL NEWS")
    else:
        print("Prediction : FAKE NEWS")

    print(f"Confidence : {confidence:.2f}%")
    print("============================")