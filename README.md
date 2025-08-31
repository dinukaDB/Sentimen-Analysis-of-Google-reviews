# Google Review Classification - Sentiment Analysis Web App

This is a Flask-based web application developed as part of my undergraduate research project.  
The system performs **sentiment analysis on Google Reviews** using classical machine learning algorithms to classify reviews as **positive** or **negative**, and determine the overall favorability of an attraction.

---

## 🚀 Features
- Analyze Google Reviews to classify sentiment (Positive / Negative)  
- Preprocessing pipeline for text cleaning and tokenization  
- Sentiment classification using multiple ML models (Logistic Regression, Naive Bayes, SVM, Decision Trees, Random Forest)  
- Model comparison and evaluation metrics (accuracy, precision, recall, F1-score, AUC)  
- Simple and interactive Flask web interface  

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask  
- **Machine Learning:** scikit-learn, pandas, numpy  
- **Frontend:** HTML, CSS, Bootstrap
- **Visualization:** Matplotlib  

---

## 📂 Project Structure

models/ # Trained ML models
static/ # CSS, JS, images
templates/ # HTML templates
Trained model & dataset/ # Dataset and saved models
venv/ # Virtual environment
app.py # Main Flask application
utils.py # Helper functions for preprocessing, model loading

## ⚙️ Installation & Setup

python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
python app.py