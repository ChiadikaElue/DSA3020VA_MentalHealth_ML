import os
import re
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, session
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production

# Initialize preprocessing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and preprocess text input"""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

def compute_confidence(model, text_vector, prediction):
    """Robustly compute confidence: try decision_function -> predict_proba -> fallback"""
    try:
        scores = model.decision_function(text_vector)
        # Extract scalar from potentially multi-dimensional array
        if hasattr(scores, 'ndim') and getattr(scores, 'ndim') > 0:
            raw = scores[0]
            if hasattr(raw, '__iter__'):
                try:
                    idx = list(model.classes_).index(prediction)
                    raw = raw[idx]
                except Exception:
                    raw = float(raw[0])
            raw = float(raw)
        else:
            raw = float(scores)
        return max(0, min(100, raw * 10 + 50))
    except Exception:
        try:
            proba = model.predict_proba(text_vector)[0]
            if hasattr(model, 'classes_'):
                idx = list(model.classes_).index(prediction)
                return round(float(proba[idx]) * 100, 2)
            return round(float(max(proba)) * 100, 2)
        except Exception:
            return 0.0

# Load models
try:
    model = joblib.load('models/model.pkl')
    vectorizer = joblib.load('models/vectorizer.pkl')
    print("✅ Models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    model = None
    vectorizer = None

@app.route('/')
def home():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle text prediction"""
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        # Get text from form
        text = request.form.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Store in session for results page
        session['original_text'] = text
        
        # Preprocess
        cleaned_text = preprocess_text(text)
        
        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        confidence = compute_confidence(model, text_vector, prediction)
        
        # Get keywords that contributed to decision
        try:
            feature_names = vectorizer.get_feature_names_out()
        except Exception:
            feature_names = vectorizer.get_feature_names()
        try:
            coefficients = model.coef_[0]
        except Exception:
            coefficients = None
        
        # Get top 10 words for each class
        anxiety_keywords = []
        normal_keywords = []
        
        # Get words present in the text
        text_words = set(cleaned_text.split())
        
        for word in text_words:
            if word in feature_names:
                idx = list(feature_names).index(word)
                if coefficients is not None:
                    weight = coefficients[idx]
                    if weight > 0.1:  # Contributes to anxiety
                        anxiety_keywords.append((word, round(float(weight), 3)))
                    elif weight < -0.1:  # Contributes to normal
                        normal_keywords.append((word, round(float(weight), 3)))
        
        # Sort by absolute weight
        anxiety_keywords.sort(key=lambda x: abs(x[1]), reverse=True)
        normal_keywords.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return render_template('results.html',
                             text=text,
                             prediction=prediction,
                             confidence=round(confidence, 2),
                             anxiety_keywords=anxiety_keywords[:10],
                             normal_keywords=normal_keywords[:10])
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded properly'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Preprocess
        cleaned_text = preprocess_text(text)
        
        # Vectorize
        text_vector = vectorizer.transform([cleaned_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        confidence = compute_confidence(model, text_vector, prediction)
        
        return jsonify({
            'text': text,
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/sample_texts')
def get_sample_texts():
    """Return sample texts for testing"""
    samples = {
        'anxiety': [
            "I feel so restless and can't sleep, my heart is racing for no reason.",
            "I'm constantly worried about everything, even small things make me anxious.",
            "Why am I so nervous all the time? I can't focus on anything.",
            "I had another panic attack today, I couldn't breathe properly.",
            "My mind won't stop thinking about worst-case scenarios."
        ],
        'normal': [
            "Had a great day at work today, completed all my tasks on time.",
            "Going out for dinner with friends this evening, looking forward to it.",
            "The weather is beautiful today, perfect for a walk in the park.",
            "Just finished reading an interesting book, would recommend it.",
            "Planning my weekend trip, excited to visit new places."
        ]
    }
    return jsonify(samples)

if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
