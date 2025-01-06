from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import re
from datetime import datetime

app = Flask(__name__)

class TransactionClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.is_trained = False
        
        # Initial training data
        self.training_data = [
            {"description": "WELCOME SUPER SHOPEE UPI", "amount": 40.00, "category": "Groceries"},
            {"description": "DAILY NEEDS STORE UPI", "amount": 150.00, "category": "Groceries"},
            {"description": "RELIANCE FRESH UPI", "amount": 3000.00, "category": "Groceries"},
            {"description": "SWIGGY FOOD UPI", "amount": 75.00, "category": "Food & Dining"},
            {"description": "ZOMATO FOOD UPI", "amount": 100.00, "category": "Food & Dining"},
            {"description": "AMAZON RETAIL UPI", "amount": 899.00, "category": "Shopping"},
            {"description": "DECATHLON SPORTS UPI", "amount": 1200.00, "category": "Shopping"},
            {"description": "NETFLIX DIGITAL UPI", "amount": 49.00, "category": "Entertainment"},
            {"description": "PVR CINEMAS UPI", "amount": 500.00, "category": "Entertainment"},
            {"description": "UBER INDIA UPI", "amount": 200.00, "category": "Transportation"},
            {"description": "OLA CABS UPI", "amount": 800.00, "category": "Transportation"},
            {"description": "APOLLO PHARMACY UPI", "amount": 1500.00, "category": "Healthcare"},
            {"description": "MAX HOSPITAL UPI", "amount": 2500.00, "category": "Healthcare"},
            {"description": "AIRTEL PREPAID UPI", "amount": 2000.00, "category": "Utilities"},
            {"description": "HDFC HOME LOAN UPI", "amount": 25000.00, "category": "Housing"},
        ]

    def preprocess_text(self, text):
        """Clean and standardize transaction descriptions"""
        text = text.lower()
        text = re.sub(r'ref no \d+', '', text)
        text = re.sub(r'a/c\w+', '', text)
        text = re.sub(r'rs\.|inr|credited to|debited|via|and', '', text)
        text = re.sub(r'\d+\.?\d*', '', text)
        return ' '.join(text.split())

    def extract_amount(self, description):
        """Extract amount from transaction description"""
        match = re.search(r'Rs\.(\d+\.?\d*)', description)
        if match:
            return float(match.group(1))
        return None

    def train(self, additional_data=None):
        """Train the classifier with optional additional training data"""
        training_data = self.training_data.copy()
        if additional_data:
            training_data.extend(additional_data)
            
        df = pd.DataFrame(training_data)
        
        # Prepare features
        X_text = self.vectorizer.fit_transform(df['description'].apply(self.preprocess_text))
        X_amount = df[['amount']].values
        X = np.hstack([X_text.toarray(), X_amount])
        
        # Train classifier
        self.classifier.fit(X, df['category'])
        self.is_trained = True
        
        return self.classifier.score(X, df['category'])

    def predict(self, description, amount=None):
        """Predict category for a single transaction"""
        if not self.is_trained:
            self.train()
        
        if amount is None:
            amount = self.extract_amount(description)
            if amount is None:
                raise ValueError("Could not extract amount from description and no amount provided")
        
        X_text = self.vectorizer.transform([self.preprocess_text(description)])
        X_amount = np.array([[amount]])
        X = np.hstack([X_text.toarray(), X_amount])
        
        return {
            'category': self.classifier.predict(X)[0],
            'probabilities': dict(zip(
                self.classifier.classes_,
                self.classifier.predict_proba(X)[0].tolist()
            ))
        }

# Initialize classifier
classifier = TransactionClassifier()
classifier.train()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/train', methods=['POST'])
def train():
    try:
        data = request.json
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No training data provided'}), 400
            
        accuracy = classifier.train(data['transactions'])
        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'message': f'Model trained successfully with accuracy: {accuracy:.2%}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/classify', methods=['POST'])
def classify():
    try:
        data = request.json
        if not data or 'description' not in data:
            return jsonify({'error': 'No transaction description provided'}), 400
            
        amount = data.get('amount')
        if amount is not None:
            amount = float(amount)
            
        result = classifier.predict(data['description'], amount)
        return jsonify({
            'success': True,
            'category': result['category'],
            'confidence_scores': result['probabilities']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bulk-classify', methods=['POST'])
def bulk_classify():
    try:
        data = request.json
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions provided'}), 400
            
        results = []
        for trans in data['transactions']:
            description = trans.get('description', '')
            amount = trans.get('amount', classifier.extract_amount(description))
            
            try:
                prediction = classifier.predict(description, amount)
                results.append({
                    'description': description,
                    'amount': amount,
                    'predicted_category': prediction['category'],
                    'confidence_scores': prediction['probabilities']
                })
            except Exception as e:
                results.append({
                    'description': description,
                    'amount': amount,
                    'error': str(e)
                })
                
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)