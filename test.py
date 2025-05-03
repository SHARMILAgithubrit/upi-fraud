import tkinter as tk
from tkinter import messagebox
import pickle

# Load model and vectorizer
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Function to predict URL type
def predict_url():
    url = entry.get()
    if not url.strip():
        messagebox.showwarning("Input Error", "Please enter a URL.")
        return

    features = vectorizer.transform([url])
    prediction = model.predict(features)[0]

    if prediction == 0:
        result_label.config(text="Genuine ✅", fg="green")
    else:
        result_label.config(text="Fraud 🚨", fg="red")

# GUI setup
root = tk.Tk()
root.title("UPI Fraud URL Detector")
root.geometry("400x250")
root.resizable(False, False)

title_label = tk.Label(root, text="UPI Fraud URL Detection", font=("Helvetica", 16, "bold"))
title_label.pack(pady=10)

entry_label = tk.Label(root, text="Enter URL:")
entry_label.pack()

entry = tk.Entry(root, width=50)
entry.pack(pady=5)

predict_btn = tk.Button(root, text="Predict", command=predict_url, bg="#007acc", fg="white", font=("Helvetica", 12))
predict_btn.pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 14))
result_label.pack(pady=10)

root.mainloop()
