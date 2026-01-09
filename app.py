from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from datetime import datetime

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
vectorizer = None
model_trained = False

# Severity level descriptions
SEVERITY_DESCRIPTIONS = {
    1: {
        'name': 'Critical',
        'description': 'Fatalities, vessel loss, explosions, major oil spills',
        'color': '#dc3545',
        'icon': '🚨'
    },
    2: {
        'name': 'Serious',
        'description': 'Serious injuries, major damage, evacuations, significant fires',
        'color': '#fd7e14',
        'icon': '⚠️'
    },
    3: {
        'name': 'Moderate',
        'description': 'Collisions, groundings, engine failures, moderate injuries',
        'color': '#ffc107',
        'icon': '⚡'
    },
    4: {
        'name': 'Minor',
        'description': 'Minor injuries, slight damage, first aid cases',
        'color': '#17a2b8',
        'icon': '📋'
    },
    5: {
        'name': 'Administrative',
        'description': 'Regulatory non-compliance, training exercises, inspections',
        'color': '#6c757d',
        'icon': '📝'
    },
    6: {
        'name': 'Unknown',
        'description': 'Unable to classify based on provided information',
        'color': '#868e96',
        'icon': '❓'
    }
}

def load_or_train_model():
    """Load existing model or train a new one"""
    global model, vectorizer, model_trained
    
    # Try to load saved model
    if os.path.exists('maritime_model.pkl') and os.path.exists('tfidf_vectorizer.pkl'):
        try:
            print("Loading existing model...")
            model = joblib.load('maritime_model.pkl')
            vectorizer = joblib.load('tfidf_vectorizer.pkl')
            model_trained = True
            print("Model loaded successfully!")
            return
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Will train a new model...")
    
    # Train new model
    print("Training new model...")
    try:
        # Load dataset
        df = pd.read_csv('classified_dataset.csv')
        
        # Filter out unknown severity (6)
        df = df[df['Severity'].isin([1, 2, 3, 4, 5])].copy()
        
        # Combine text fields
        df['Text'] = df['Short_Description'].fillna('') + ' ' + df['Description'].fillna('')
        
        # Prepare features and target
        X = df['Text']
        y = df['Severity']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create pipeline
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')),
            ('clf', GradientBoostingClassifier(n_estimators=50, random_state=42, learning_rate=0.1, max_depth=6))
        ])
        
        # Train model
        print("Fitting model...")
        pipeline.fit(X_train, y_train)
        
        # Extract components
        vectorizer = pipeline.named_steps['tfidf']
        model = pipeline.named_steps['clf']
        
        # Save model and vectorizer
        joblib.dump(model, 'maritime_model.pkl')
        joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
        
        model_trained = True
        print("Model trained and saved successfully!")
        
        # Calculate accuracy
        train_score = model.score(vectorizer.transform(X_train), y_train)
        test_score = model.score(vectorizer.transform(X_test), y_test)
        print(f"Training accuracy: {train_score:.4f}")
        print(f"Test accuracy: {test_score:.4f}")
        
    except FileNotFoundError:
        print("Dataset file 'classified_dataset.csv' not found!")
        print("Please ensure the dataset is available.")
        model_trained = False
    except Exception as e:
        print(f"Error training model: {e}")
        model_trained = False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html', severities=SEVERITY_DESCRIPTIONS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Please provide incident description'
            }), 400
        
        if not model_trained:
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please check server logs.'
            }), 500
        
        # Preprocess text
        combined_text = text
        
        # Transform using vectorizer
        text_vectorized = vectorizer.transform([combined_text])
        
        # Predict
        prediction = model.predict(text_vectorized)[0]
        
        # Get severity info
        severity_info = SEVERITY_DESCRIPTIONS.get(prediction, SEVERITY_DESCRIPTIONS[6])
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'severity_name': severity_info['name'],
            'severity_description': severity_info['description'],
            'severity_color': severity_info['color'],
            'severity_icon': severity_info['icon']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_trained,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Maritime Incident Severity Classification System")
    print("=" * 60)
    load_or_train_model()
    print("=" * 60)
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)

