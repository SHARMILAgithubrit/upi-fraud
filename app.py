from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    if not url or not url.strip():
        return jsonify({'error': 'Please enter a URL.'})
    
    features = vectorizer.transform([url])
    prediction = model.predict(features)[0]
    
    if prediction == 0:
        return jsonify({'result': 'Genuine ✅', 'class': 'genuine'})
    else:
        return jsonify({'result': 'Fraud 🚨', 'class': 'fraud'})

if __name__ == '__main__':
    app.run(debug=True)