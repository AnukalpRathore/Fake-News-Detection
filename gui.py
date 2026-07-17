import tkinter as tk
from tkinter import messagebox
import pickle
from datetime import datetime

# Load model and vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Prediction function
def predict_news():
    news = text_box.get("1.0", tk.END).strip()

    if news == "":
        messagebox.showwarning("Warning", "Please enter news.")
        return

    vector = vectorizer.transform([news])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0]) * 100

    if prediction == 1:
        result.config(text="✅ REAL NEWS", fg="green")
    else:
        result.config(text="❌ FAKE NEWS", fg="red")

    confidence_label.config(text=f"Confidence : {confidence:.2f}%")
    date_label.config(text=datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"))

# Clear function
def clear_text():
    text_box.delete("1.0", tk.END)
    result.config(text="")
    confidence_label.config(text="")
    date_label.config(text="")

# GUI
root = tk.Tk()
root.title("Fake News Detection")
root.geometry("700x550")
root.configure(bg="#f2f2f2")

title = tk.Label(
    root,
    text="FAKE NEWS DETECTION",
    font=("Arial", 20, "bold"),
    bg="#f2f2f2",
    fg="blue"
)
title.pack(pady=10)

text_box = tk.Text(root, height=12, width=75, font=("Arial", 11))
text_box.pack(pady=10)

predict_btn = tk.Button(
    root,
    text="Predict",
    command=predict_news,
    bg="green",
    fg="white",
    font=("Arial", 12, "bold")
)
predict_btn.pack(pady=5)

clear_btn = tk.Button(
    root,
    text="Clear",
    command=clear_text,
    bg="red",
    fg="white",
    font=("Arial", 12, "bold")
)
clear_btn.pack(pady=5)

result = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#f2f2f2")
result.pack(pady=10)

confidence_label = tk.Label(root, text="", font=("Arial", 13), bg="#f2f2f2")
confidence_label.pack()

date_label = tk.Label(root, text="", font=("Arial", 12), bg="#f2f2f2")
date_label.pack()

root.mainloop()