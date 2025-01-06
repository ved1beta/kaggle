from upi import EnhancedBankStatementCategorizer
import pandas as pd
from datetime import datetime

def run_categorizer_demo():
    # Initialize the categorizer
    print("Initializing categorizer...")
    categorizer = EnhancedBankStatementCategorizer()

    # Step 1: Train the model
    print("\nStep 1: Training the model...")
    try:
        accuracy = categorizer.train('training_data.csv')
        print(f"Training completed with accuracy: {accuracy:.2%}")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return

    # Step 2: Create a sample transaction file to categorize
    print("\nStep 2: Creating sample transactions...")
    sample_transactions = pd.DataFrame({
        'date': ['2024-12-10', '2024-12-11', '2024-12-12'],
        'description': [
            'Rs.40.00 debited A/cXX0059 and credited to WELCOME SUPER SHOPEE via UPI Ref No 849269244940',
            'Rs.499.00 debited A/cXX0059 and credited to SWIGGY FOOD via UPI Ref No 849269244941',
            'Rs.2000.00 debited A/cXX0059 and credited to APOLLO PHARMACY via UPI Ref No 849269244942'
        ],
        'amount': [40.00, 499.00, 2000.00]
    })
    sample_transactions.to_csv('transactions.csv', index=False)

    # Step 3: Categorize transactions
    print("\nStep 3: Categorizing transactions...")
    try:
        results, summary = categorizer.categorize_statement('transactions.csv')
        print("\nCategorization Results:")
        print(results[['date', 'description', 'amount', 'predicted_category']])
        
        print("\nSpending Summary:")
        print(summary)
    except Exception as e:
        print(f"Error during categorization: {str(e)}")
        return

    # Step 4: Demonstrate queries
    print("\nStep 4: Demonstrating different queries...")
    
    # Query by category
    print("\nGrocery Transactions:")
    grocery_txns = categorizer.query_transactions('category', category='Groceries')
    print(grocery_txns[['date', 'description', 'amount']] if not grocery_txns.empty else "No grocery transactions found")
    
    # Query by date range
    print("\nTransactions in December 2024:")
    date_txns = categorizer.query_transactions('date_range', 
                                             start_date='2024-12-01', 
                                             end_date='2024-12-31')
    print(date_txns[['date', 'description', 'amount', 'predicted_category']])
    
    # Query by amount range
    print("\nTransactions between Rs.100 and Rs.1000:")
    amount_txns = categorizer.query_transactions('amount_range', 
                                               min_amount=100, 
                                               max_amount=1000)
    print(amount_txns[['date', 'description', 'amount', 'predicted_category']])

    # Step 5: Get insights
    print("\nStep 5: Getting spending insights...")
    insights = categorizer.get_spending_insights()
    print("\nTop spending categories:")
    print(insights['top_categories'])
    print("\nAverage transaction amount:", round(insights['average_transaction'], 2))
    print("\nLargest transaction:")
    print(insights['largest_transaction'][['date', 'description', 'amount', 'predicted_category']])

if __name__ == "__main__":
    run_categorizer_demo()