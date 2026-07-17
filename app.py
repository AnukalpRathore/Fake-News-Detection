from flask import Flask, render_template, request
from datetime import datetime
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

accuracy = 98.7
history = []

@app.route("/", methods=["GET", "POST"])
def home():
    result = ""
    confidence = ""
    time = ""

    if request.method == "POST":
        news = request.form["news"]

        data = vectorizer.transform([news])
        prediction = model.predict(data)[0]
        probability = max(model.predict_proba(data)[0]) * 100

        if prediction == 1:
            result = "REAL NEWS ✅"
        else:
            result = "FAKE NEWS ❌"

        confidence = round(probability, 2)
        time = datetime.now().strftime("%d %B %Y %I:%M %p")

        history.append({
            "news": news[:40] + "..." if len(news) > 40 else news,
            "result": result,
            "time": time
        })

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        time=time,
        accuracy=accuracy,
        history=history[-5:]
    )

if __name__ == "__main__":
    app.run(debug=True)