# app.py
from flask import Flask, request, render_template, jsonify
import joblib
import os   
from utils import preprocess_text  # Import from utils module

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
model = joblib.load('models/sentiment_model.joblib')
vectorizer = joblib.load('models/vectorizer.joblib')

# Function to predict sentiment
def predict_sentiment(text):
    # Clean the input text
    cleaned_text = preprocess_text(text)
    
    # Vectorize the cleaned text
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(vectorized_text)[0]
    
    # Get confidence
    confidence = model.predict_proba(vectorized_text)[0][prediction]
    
    # Return result
    return {
        'sentiment': 'Positive' if prediction == 1 else 'Negative',
        'confidence': f"{confidence:.2%}"
    }

# Route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get text from form
    text = request.form.get('text', '')
    
    # Make prediction
    result = predict_sentiment(text)
    
    # Return result to template
    return render_template('index.html', 
                          text=text, 
                          sentiment=result['sentiment'],
                          confidence=result['confidence'])

# Create templates directory
os.makedirs('templates', exist_ok=True)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)