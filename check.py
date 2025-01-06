import pandas as pd

try:
    # Try to read the file
    print("Attempting to read training_data.csv...")
    df = pd.read_csv('training_data.csv')
    
    # Print the column names
    print("\nColumns in the file:")
    print(df.columns.tolist())
    
    # Print the first few rows
    print("\nFirst few rows of data:")
    print(df.head())
    
except Exception as e:
    print(f"Error reading file: {str(e)}")