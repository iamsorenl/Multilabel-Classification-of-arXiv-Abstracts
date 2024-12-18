import argparse
import json
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from lr_onevsrest import train_logistic_regression
from sgd_model import sgd_model
from stack_classifier import stacked_classifier_model
from knn_model import knn_model
from hierarchical_model import hierarchical_model

def reduce_labels(data, threshold):
    # Flatten the lists of labels (assumes 'terms' column contains lists)
    all_labels = data['terms'].explode()  # Expanding lists into individual elements
    label_counts = all_labels.value_counts()  # Count occurrences of each label

    # Filter out labels with fewer than the threshold number of occurrences
    labels_to_keep = label_counts[label_counts >= threshold].index.tolist()
    print(f"\nReducing labels from {len(label_counts)} to {len(labels_to_keep)} based on a threshold of {threshold} occurrences.")
    reduced_data = data[data['terms'].apply(lambda x: all(label in labels_to_keep for label in x))]

    return reduced_data

# this version labels rare labels as unknown
'''def reduce_labels(data, threshold):
    # Flatten the lists of labels (assumes 'terms' column contains lists)
    all_labels = data['terms'].explode()  # Expanding lists into individual elements
    label_counts = all_labels.value_counts()  # Count occurrences of each label

    # Identify labels to keep based on the threshold
    labels_to_keep = label_counts[label_counts >= threshold].index.tolist()
    print(f"\nReducing labels from {len(label_counts)} to {len(labels_to_keep)} based on a threshold of {threshold} occurrences.")

    # Replace labels not in labels_to_keep with 'unknown'
    def replace_labels_with_unknown(label_list):
        return [label if label in labels_to_keep else 'unknown' for label in label_list]

    # Update the 'terms' column with replaced labels
    updated_data = data.copy()
    updated_data['terms'] = updated_data['terms'].apply(replace_labels_with_unknown)

    return updated_data'''

# this version goes further and removes rows where the majority of labels are unknown
'''def reduce_labels(data, threshold):
    # Flatten the lists of labels (assumes 'terms' column contains lists)
    all_labels = data['terms'].explode()  # Expanding lists into individual elements
    label_counts = all_labels.value_counts()  # Count occurrences of each label

    # Identify labels to keep based on the threshold
    labels_to_keep = label_counts[label_counts >= threshold].index.tolist()
    print(f"\nReducing labels from {len(label_counts)} to {len(labels_to_keep)} based on a threshold of {threshold} occurrences.")

    # Replace labels not in labels_to_keep with 'unknown'
    def replace_labels_with_unknown(label_list):
        return [label if label in labels_to_keep else 'unknown' for label in label_list]

    # Update the 'terms' column with replaced labels
    updated_data = data.copy()
    updated_data['terms'] = updated_data['terms'].apply(replace_labels_with_unknown)

    # Remove rows where the majority of labels are 'unknown'
    def is_majority_unknown(label_list):
        unknown_count = sum(label == 'unknown' for label in label_list)
        return unknown_count > len(label_list) / 2

    updated_data = updated_data[~updated_data['terms'].apply(is_majority_unknown)]

    return updated_data'''

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

def main(data, outfile, plot, model, threshold):
    # Load the data
    with open(data, 'r') as f:
        arxiv_data = json.load(f)

    # Split the data
    train, val, test = split_data(arxiv_data)

    # Reduce labels based on frequency
    if threshold > 0:
        print(f"Length of training data before reducing labels: {len(train)}")
        train = reduce_labels(train, threshold)
        print(f"Length of training data after reducing labels: {len(train)}")

    # Plot the data distributions and print counts
    if plot:
        plot_label_distribution(train, 'Training Data')
        plot_label_distribution(val, 'Validation Data')
        plot_label_distribution(test, 'Test Data')

    # Train the Naive Bayes model
    if model == 'lr':
        train_logistic_regression(train, val, test, outfile)
    elif model == 'sgd':
        sgd_model(train, val, test, outfile)
    elif model == 'stack':
        stacked_classifier_model(train, val, test, outfile)
    elif model == 'knn':
        knn_model(train, val, test, outfile)
    elif model == 'hier':
        hierarchical_model(train, val, test, outfile)
    else:
        print("Invalid model. Please choose 'lr', 'sgd', 'stack', 'knn', or 'hier'.")

if __name__ == "__main__":
    # example use: python homework3.py --data arxiv_data.json --output results.txt --plot True --model hier --threshold 0
    parser = argparse.ArgumentParser(description="Process arXiv data and output results.")
    parser.add_argument('--data', type=str, default="arxiv_data.json", help='Path to the input data file (e.g., arxiv_data.json)')
    parser.add_argument('--output', type=str, default="results.txt", help='Path to the output results file (e.g., results.txt)')
    parser.add_argument('--plot', type=bool, default=False, help='Whether to plot the data (True or False)')
    parser.add_argument('--model', type=str, default='hier', help='Models to use: lr (for Logistic Regression with OneVsRest), sgd (for MultiOutputClassifier with SGD), stack (for StackingClassifier), knn (for KNeighborsClassifier), hier (for hierarchical model)')
    parser.add_argument('--threshold', type=int, default=0, help='Threshold for reducing labels based on frequency')

    args = parser.parse_args()
    main(args.data, args.output, args.plot, args.model, args.threshold)
