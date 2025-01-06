import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy

class TransactionCategorizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.text_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        self.merchant_cache = {}
        self.is_trained = False
        self.features_fitted = False

    def preprocess_description(self, description):
        text = description.lower()
        text = ' '.join(text.split())
        return text

    def prepare_features(self, transactions_df, training=False):
        # Prepare text features
        descriptions = transactions_df['description'].apply(self.preprocess_description)
        
        # During training, fit_transform the vectorizer
        # During prediction, just transform using the fitted vectorizer
        if training and not self.features_fitted:
            text_features = self.text_vectorizer.fit_transform(descriptions)
            self.features_fitted = True
        else:
            text_features = self.text_vectorizer.transform(descriptions)
        
        # Prepare amount features
        amount_features = transactions_df[['amount']].values
        
        # Combine features
        return np.hstack([text_features.toarray(), amount_features])

    def train(self, transactions_df):
        # Prepare features and target
        X = self.prepare_features(transactions_df, training=True)
        y = transactions_df['category']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Return accuracy score
        accuracy = self.classifier.score(X_test, y_test)
        return accuracy

    def predict(self, transaction):
        if not self.is_trained:
            raise Exception("Model needs to be trained first!")
        
        # Convert single transaction to DataFrame
        if isinstance(transaction, dict):
            transaction = pd.DataFrame([transaction])
        
        # Prepare features using the fitted vectorizer
        X = self.prepare_features(transaction, training=False)
        
        # Make prediction
        return self.classifier.predict(X)[0]

def generate_sample_transactions(num_transactions=1000):
    # [Previous generate_sample_transactions function remains the same]
    merchants = {
        'NETFLIX': {
            'category': 'Entertainment',
            'amount_range': (9.99, 19.99),
            'subscription': True
        },
        'WALMART': {
            'category': 'Groceries',
            'amount_range': (30, 200),
            'subscription': False
        },
        'SHELL OIL': {
            'category': 'Transportation',
            'amount_range': (20, 80),
            'subscription': False
        },
        'AMC MOVIES': {
            'category': 'Entertainment',
            'amount_range': (10, 50),
            'subscription': False
        },
        'PLANET FITNESS': {
            'category': 'Health',
            'amount_range': (10, 50),
            'subscription': True
        },
        'UBER': {
            'category': 'Transportation',
            'amount_range': (10, 40),
            'subscription': False
        },
        'TRADER JOES': {
            'category': 'Groceries',
            'amount_range': (20, 150),
            'subscription': False
        },
        'AT&T WIRELESS': {
            'category': 'Utilities',
            'amount_range': (50, 150),
            'subscription': True
        },
        'AMAZON PRIME': {
            'category': 'Shopping',
            'amount_range': (10, 200),
            'subscription': False
        },
        'STARBUCKS': {
            'category': 'Dining',
            'amount_range': (3, 20),
            'subscription': False
        }
    }

    transactions = []
    start_date = datetime(2024, 1, 1)
    
    for _ in range(num_transactions):
        merchant_name = random.choice(list(merchants.keys()))
        merchant_info = merchants[merchant_name]
        
        transaction = {
            'date': start_date + timedelta(days=random.randint(0, 180)),
            'description': f"{merchant_name} #{random.randint(100, 999)}",
            'amount': round(random.uniform(*merchant_info['amount_range']), 2),
            'category': merchant_info['category']
        }
        
        transactions.append(transaction)
    
    for merchant_name, info in merchants.items():
        if info['subscription']:
            for month in range(6):
                transaction = {
                    'date': start_date + timedelta(days=30 * month),
                    'description': f"{merchant_name} SUBSCRIPTION",
                    'amount': round(info['amount_range'][0], 2),
                    'category': info['category']
                }
                transactions.append(transaction)
    
    df = pd.DataFrame(transactions)
    return df.sort_values('date')

def main():
    # Generate sample data
    print("Generating sample transactions...")
    transactions_df = generate_sample_transactions(1000)
    print(f"Generated {len(transactions_df)} transactions")
    
    # Initialize and train the categorizer
    print("\nInitializing categorizer...")
    categorizer = TransactionCategorizer()
    
    print("Training model...")
    accuracy = categorizer.train(transactions_df)
    print(f"Model trained with accuracy: {accuracy:.2%}")
    
    # Test some predictions
    print("\nTesting predictions...")
    test_transactions = [
        {
            'date': datetime(2024, 1, 15),
            'description': 'WALMART #456',
            'amount': 123.45
        },
        {
            'date': datetime(2024, 1, 16),
            'description': 'NETFLIX MONTHLY',
            'amount': 14.99
        },
        {
            'date': datetime(2024, 1, 17),
            'description': 'UBER TRIP',
            'amount': 25.50
        }
    ]
    
    for transaction in test_transactions:
        category = categorizer.predict(transaction)
        print(f"\nTransaction: {transaction['description']}")
        print(f"Amount: ${transaction['amount']}")
        print(f"Predicted category: {category}")
    
    # Display some statistics
    print("\nTransaction Statistics:")
    print("\nTransactions by Category:")
    print(transactions_df['category'].value_counts())
    
    print("\nAverage Amount by Category:")
    print(transactions_df.groupby('category')['amount'].mean().round(2))

if __name__ == "__main__":
    main()