import argparse
import json
from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(data):
    df = pd.DataFrame(data)
    
    # 70% train, 15% validation, 15% test split
    train, valtest = train_test_split(df, test_size=0.30, random_state=1234)
    val, test = train_test_split(valtest, test_size=0.50, random_state=1234)
    
    return train, val, test

def main(data, outfile):
    with open(data, 'r') as f:
        arxiv_data = json.load(f)

    # Split the data
    train, val, test = split_data(arxiv_data)

    # Example output to see the split sizes
    print(f"Train size: {len(train)}")
    print(f"Validation size: {len(val)}")
    print(f"Test size: {len(test)}")

if __name__ == "__main__":
    # example use: python homework3.py --data arxiv_data.json --output results.txt
    parser = argparse.ArgumentParser(description="Process arXiv data and output results.")
    parser.add_argument('--data', required=True, help='Path to the input data file (e.g., arxiv_data.json)')
    parser.add_argument('--output', required=True, help='Path to the output results file (e.g., results.txt)')
    
    args = parser.parse_args()
    main(args.data, args.output)