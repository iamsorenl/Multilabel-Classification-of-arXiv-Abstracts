import argparse
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from lr_onevsrest import train_logistic_regression

def split_data(data):
    df = pd.DataFrame(data)
    
    # 70% train, 15% validation, 15% test split
    train, valtest = train_test_split(df, test_size=0.30, random_state=1234)
    val, test = train_test_split(valtest, test_size=0.50, random_state=1234)
    
    return train, val, test

def plot_label_distribution(df, title):
    # Flatten the lists of labels (assumes 'terms' column contains lists)
    all_labels = df['terms'].explode()  # Expanding lists into individual elements
    label_counts = all_labels.value_counts()  # Count occurrences of each label

    # Print the labels and their counts to the terminal
    print(f"\nLabel counts for in {title}:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    print(f"Length of {title}: {len(df)}")

    # Plotting
    plt.figure(figsize=(12, 6))
    label_counts.plot(kind='bar')
    plt.title("Label Distribution in " + title)
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def main(data, outfile, plot, model):
    # Load the data
    with open(data, 'r') as f:
        arxiv_data = json.load(f)

    # Split the data
    train, val, test = split_data(arxiv_data)

    # Plot the data distributions and print counts
    if plot:
        plot_label_distribution(train, 'Training Data')
        plot_label_distribution(val, 'Validation Data')
        plot_label_distribution(test, 'Test Data')

    # Train the Naive Bayes model
    if model == 'lr':
        train_logistic_regression(train, val, test, outfile)


if __name__ == "__main__":
    # example use: python homework3.py --data arxiv_data.json --output results.txt --plot True
    parser = argparse.ArgumentParser(description="Process arXiv data and output results.")
    parser.add_argument('--data', required=True, help='Path to the input data file (e.g., arxiv_data.json)')
    parser.add_argument('--output', required=True, help='Path to the output results file (e.g., results.txt)')
    parser.add_argument('--plot', type=bool, default=False, help='Whether to plot the data (True or False)')
    parser.add_argument('--model', type=str, default='lr', help='Models to use: lr (for Logistic Regression with OneVsRest)')
    
    args = parser.parse_args()
    main(args.data, args.output, args.plot, args.model)
