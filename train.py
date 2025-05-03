import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("upi_fraud_binary.csv")  # Should contain 'url' and 'label'

# Sample for fast testing (remove for full training)
df = df.sample(10000, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=2000)
X = vectorizer.fit_transform(df['url'])
y = df['label']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=-1),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM": SVC(kernel='linear')
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔍 Training: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("🧾 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save the Random Forest model and vectorizer
    if name == "Random Forest":
        with open("random_forest_model.pkl", "wb") as f:
            pickle.dump(model, f)
        with open("tfidf_vectorizer.pkl", "wb") as f:
            pickle.dump(vectorizer, f)
        print("💾 Saved: random_forest_model.pkl and tfidf_vectorizer.pkl")
