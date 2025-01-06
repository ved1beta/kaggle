import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
import csv

class BankStatementCategorizer:
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
        self.is_trained = False
        self.features_fitted = False
        
        # Common column names in bank statements
        self.date_columns = ['date', 'transaction date', 'posted date', 'trans date']
        self.description_columns = ['description', 'transaction', 'details', 'merchant', 'name']
        self.amount_columns = ['amount', 'transaction amount', 'debit', 'credit']

    def read_bank_statement(self, file_path):
        """
        Read bank statement from CSV file and standardize the format
        """
        try:
            # First, try to read with pandas
            df = pd.read_csv(file_path)
        except:
            # If fails, try to detect delimiter
            with open(file_path, 'r') as file:
                dialect = csv.Sniffer().sniff(file.read(1024))
                df = pd.read_csv(file_path, sep=dialect.delimiter)

        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()

        # Identify key columns
        date_col = next((col for col in df.columns if col in self.date_columns), None)
        desc_col = next((col for col in df.columns if col in self.description_columns), None)
        amount_col = next((col for col in df.columns if col in self.amount_columns), None)

        if not all([date_col, desc_col, amount_col]):
            raise ValueError("Could not identify required columns in the bank statement")

        # Create standardized DataFrame
        standardized_df = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'description': df[desc_col].astype(str),
            'amount': df[amount_col].astype(float)
        })

        return standardized_df

    def preprocess_description(self, description):
        """Clean and standardize transaction descriptions"""
        text = description.lower()
        text = ' '.join(text.split())
        return text

    def prepare_features(self, transactions_df, training=False):
        """Prepare features for model"""
        descriptions = transactions_df['description'].apply(self.preprocess_description)
        
        if training and not self.features_fitted:
            text_features = self.text_vectorizer.fit_transform(descriptions)
            self.features_fitted = True
        else:
            text_features = self.text_vectorizer.transform(descriptions)
        
        amount_features = transactions_df[['amount']].values
        return np.hstack([text_features.toarray(), amount_features])

    def train(self, labeled_data_path):
        """Train the model using labeled transaction data"""
        # Read labeled training data
        training_df = self.read_bank_statement(labeled_data_path)
        
        if 'category' not in training_df.columns:
            raise ValueError("Training data must include 'category' column")

        # Prepare features and train
        X = self.prepare_features(training_df, training=True)
        y = training_df['category']
        
        self.classifier.fit(X, y)
        self.is_trained = True
        return self.classifier.score(X, y)

    def categorize_statement(self, statement_path, output_path=None):
        """Categorize all transactions in a bank statement"""
        if not self.is_trained:
            raise Exception("Model needs to be trained first!")

        # Read the bank statement
        transactions_df = self.read_bank_statement(statement_path)
        
        # Prepare features and predict
        X = self.prepare_features(transactions_df)
        predictions = self.classifier.predict(X)
        
        # Add predictions to DataFrame
        transactions_df['predicted_category'] = predictions
        
        # Calculate spending by category
        category_spending = transactions_df.groupby('predicted_category')['amount'].agg([
            'count',
            'sum',
            'mean'
        ]).round(2)

        # Save categorized transactions if output path provided
        if output_path:
            transactions_df.to_csv(output_path, index=False)
            
        return transactions_df, category_spending

def main():
    """Example usage of the BankStatementCategorizer"""
    categorizer = BankStatementCategorizer()
    
    # Example of how to use with real data:
    print("To use this system with your bank statement:")
    print("\n1. First, train the model with labeled data:")
    print("   categorizer.train('labeled_transactions.csv')")
    print("\n2. Then categorize a new bank statement:")
    print("   results, summary = categorizer.categorize_statement('bank_statement.csv')")
    
    # Example of required CSV format for training data:
    print("\nRequired CSV format for training data:")
    example_df = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'description': ['WALMART GROCERY', 'NETFLIX SUBSCRIPTION'],
        'amount': [50.25, 14.99],
        'category': ['Groceries', 'Entertainment']
    })
    print("\nExample training data format:")
    print(example_df.to_string())
    
    # Example of required CSV format for bank statements:
    print("\nRequired CSV format for bank statements:")
    example_statement = pd.DataFrame({
        'date': ['2024-01-01', '2024-01-02'],
        'description': ['WALMART GROCERY', 'NETFLIX SUBSCRIPTION'],
        'amount': [50.25, 14.99]
    })
    print("\nExample bank statement format:")
    print(example_statement.to_string())

if __name__ == "__main__":
    main()