import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import spacy
import csv
from datetime import datetime, timedelta

class EnhancedBankStatementCategorizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.text_vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),  # Increased to capture more context
            max_features=1500,   # Increased features
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(
            n_estimators=150,    # Increased number of trees
            random_state=42,
            min_samples_leaf=2   # Better handling of rare categories
        )
        self.is_trained = False
        self.features_fitted = False
        self.categorized_data = None
        
        # Enhanced column names for Indian bank statements
        self.date_columns = ['date', 'transaction date', 'posted date', 'trans date']
        self.description_columns = ['description', 'transaction', 'details', 'merchant', 'name', 'narration']
        self.amount_columns = ['amount', 'transaction amount', 'debit', 'credit']
        
        # Define category keywords for better classification
        self.category_keywords = {
            'Groceries': ['super', 'grocery', 'mart', 'fresh', 'store', 'kirana', 'shopee'],
            'Food & Dining': ['restaurant', 'food', 'swiggy', 'zomato', 'hotel'],
            'Shopping': ['mall', 'retail', 'shop', 'amazon', 'flipkart'],
            'Transportation': ['uber', 'ola', 'metro', 'cab', 'auto', 'fuel'],
            'Healthcare': ['hospital', 'pharmacy', 'medical', 'healthcare', 'clinic'],
            'Utilities': ['electricity', 'water', 'gas', 'broadband', 'mobile', 'recharge'],
            'Entertainment': ['movie', 'netflix', 'amazon prime', 'pvr', 'theatre'],
            'Housing': ['rent', 'maintenance', 'loan', 'emi'],
        }

    def extract_amount_from_description(self, description):
        """Extract amount from Indian bank statement description format"""
        try:
            # Look for Rs. followed by amount
            import re
            amount_match = re.search(r'Rs\.?\s*(\d+\.?\d*)', description)
            if amount_match:
                return float(amount_match.group(1))
        except:
            return None
        return None

    def read_bank_statement(self, file_path):
        """Enhanced read_bank_statement with better Indian format handling"""
        try:
            df = pd.read_csv(file_path)
        except:
            with open(file_path, 'r') as file:
                dialect = csv.Sniffer().sniff(file.read(1024))
                df = pd.read_csv(file_path, sep=dialect.delimiter)

        df.columns = df.columns.str.lower().str.strip()
        
        # Identify columns
        date_col = next((col for col in df.columns if col in self.date_columns), None)
        desc_col = next((col for col in df.columns if col in self.description_columns), None)
        amount_col = next((col for col in df.columns if col in self.amount_columns), None)

        # If amount column not found, try to extract from description
        if not amount_col and desc_col:
            df['amount'] = df[desc_col].apply(self.extract_amount_from_description)
            amount_col = 'amount'

        if not all([date_col, desc_col, amount_col]):
            raise ValueError("Could not identify required columns in the bank statement")

        standardized_df = pd.DataFrame({
            'date': pd.to_datetime(df[date_col]),
            'description': df[desc_col].astype(str),
            'amount': df[amount_col].astype(float)
        })

        return standardized_df

    def preprocess_description(self, description):
        """Enhanced preprocessing for Indian transaction descriptions"""
        text = description.lower()
        # Remove common Indian banking terms that don't help classification
        text = text.replace('a/c', '')
        text = text.replace('upi', '')
        text = text.replace('ref no', '')
        text = text.replace('credited to', '')
        text = text.replace('debited from', '')
        # Remove reference numbers
        import re
        text = re.sub(r'\d{9,}', '', text)
        text = ' '.join(text.split())
        return text

    def prepare_features(self, transactions_df, training=False):
        """Prepare features with enhanced preprocessing"""
        descriptions = transactions_df['description'].apply(self.preprocess_description)
        
        if training and not self.features_fitted:
            text_features = self.text_vectorizer.fit_transform(descriptions)
            self.features_fitted = True
        else:
            text_features = self.text_vectorizer.transform(descriptions)
        
        amount_features = transactions_df[['amount']].values
        return np.hstack([text_features.toarray(), amount_features])

    def train(self, labeled_data_path):
        """Train with enhanced features"""
        training_df = self.read_bank_statement(labeled_data_path)
        
        if 'category' not in training_df.columns:
            raise ValueError("Training data must include 'category' column")

        X = self.prepare_features(training_df, training=True)
        y = training_df['category']
        
        self.classifier.fit(X, y)
        self.is_trained = True
        return self.classifier.score(X, y)

    def categorize_statement(self, statement_path, output_path=None):
        """Categorize transactions with enhanced analysis"""
        if not self.is_trained:
            raise Exception("Model needs to be trained first!")

        transactions_df = self.read_bank_statement(statement_path)
        X = self.prepare_features(transactions_df)
        predictions = self.classifier.predict(X)
        
        transactions_df['predicted_category'] = predictions
        
        # Store categorized data for querying
        self.categorized_data = transactions_df
        
        # Enhanced spending analysis
        category_spending = transactions_df.groupby('predicted_category').agg({
            'amount': ['count', 'sum', 'mean', 'min', 'max'],
            'description': lambda x: ', '.join(x.head(3))  # Sample transactions
        }).round(2)
        
        if output_path:
            transactions_df.to_csv(output_path, index=False)
            
        return transactions_df, category_spending

    def query_transactions(self, query_type, **kwargs):
        """
        Query categorized transactions based on various criteria
        
        Parameters:
        query_type: str - Type of query ('category', 'date_range', 'amount_range', 'merchant')
        **kwargs: Additional parameters based on query type
        """
        if self.categorized_data is None:
            raise Exception("No categorized data available. Run categorize_statement first!")

        if query_type == 'category':
            category = kwargs.get('category')
            return self.categorized_data[self.categorized_data['predicted_category'] == category]
        
        elif query_type == 'date_range':
            start_date = kwargs.get('start_date')
            end_date = kwargs.get('end_date')
            mask = (self.categorized_data['date'] >= start_date) & (self.categorized_data['date'] <= end_date)
            return self.categorized_data[mask]
        
        elif query_type == 'amount_range':
            min_amount = kwargs.get('min_amount', 0)
            max_amount = kwargs.get('max_amount', float('inf'))
            mask = (self.categorized_data['amount'] >= min_amount) & (self.categorized_data['amount'] <= max_amount)
            return self.categorized_data[mask]
        
        elif query_type == 'merchant':
            merchant = kwargs.get('merchant').lower()
            return self.categorized_data[self.categorized_data['description'].str.lower().str.contains(merchant)]
        
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    def get_spending_insights(self):
        """Generate insights about spending patterns"""
        if self.categorized_data is None:
            raise Exception("No categorized data available. Run categorize_statement first!")

        insights = {
            'top_categories': self.categorized_data.groupby('predicted_category')['amount'].sum().nlargest(3),
            'average_transaction': self.categorized_data['amount'].mean(),
            'largest_transaction': self.categorized_data.nlargest(1, 'amount'),
            'monthly_trend': self.categorized_data.set_index('date').resample('M')['amount'].sum()
        }
        return insights

def main():
    """Example usage of the EnhancedBankStatementCategorizer"""
    categorizer = EnhancedBankStatementCategorizer()
    
    # Training example
    print("1. Train the model:")
    print("   categorizer.train('training_data.csv')")
    
    # Categorization example
    print("\n2. Categorize transactions:")
    print("   results, summary = categorizer.categorize_statement('transactions.csv')")
    
    # Query examples
    print("\n3. Query examples:")
    print("   # Get all grocery transactions:")
    print("   grocery_txns = categorizer.query_transactions('category', category='Groceries')")
    print("\n   # Get transactions by date range:")
    print("   date_txns = categorizer.query_transactions('date_range', start_date='2024-01-01', end_date='2024-01-31')")
    print("\n   # Get transactions by amount range:")
    print("   amount_txns = categorizer.query_transactions('amount_range', min_amount=1000, max_amount=5000)")
    print("\n   # Get transactions by merchant:")
    print("   merchant_txns = categorizer.query_transactions('merchant', merchant='SWIGGY')")
    
    # Get insights example
    print("\n4. Get spending insights:")
    print("   insights = categorizer.get_spending_insights()")

if __name__ == "__main__":
    main()