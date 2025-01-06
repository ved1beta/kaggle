import pandas as pd
from upi import EnhancedBankStatementCategorizer

def main():
    # Initialize categorizer
    print("Initializing categorizer...")
    categorizer = EnhancedBankStatementCategorizer()

    # Debug: Print the content of training data
    print("\nReading training data...")
    training_data = pd.read_csv('training_data.csv')
    print("Training data columns:", training_data.columns.tolist())
    print("Number of training examples:", len(training_data))

    # Train the model
    print("\nTraining the model...")
    try:
        accuracy = categorizer.train('training_data.csv')
        print(f"Training completed with accuracy: {accuracy:.2%}")
    except Exception as e:
        import traceback
        print(f"Error during training: {str(e)}")
        print("\nFull error traceback:")
        print(traceback.format_exc())
        return

if __name__ == "__main__":
    main()