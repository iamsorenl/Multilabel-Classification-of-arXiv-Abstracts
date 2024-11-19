import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def train_logistic_regression(train, val, test, outfile):
    # Open the output file for writing
    with open(outfile, 'w') as f:
        # Extract features and labels (assumes 'terms' and other fields exist for labels and features)
        train_labels = train['terms']
        val_labels = val['terms']
        test_labels = test['terms']

        # Combine text columns to create features (adjust as needed for your data)
        train_text = train['titles'] + " " + train['summaries']
        val_text = val['titles'] + " " + val['summaries']
        test_text = test['titles'] + " " + test['summaries']

        # Vectorize the text data using TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=30000, stop_words='english', ngram_range=(1, 2), min_df=2)
        train_features = vectorizer.fit_transform(train_text)
        val_features = vectorizer.transform(val_text)
        test_features = vectorizer.transform(test_text)

        # Scalarize the features
        scaler = StandardScaler(with_mean=False)
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        # Transform labels for multi-label classification
        mlb = MultiLabelBinarizer()
        train_labels_binary = mlb.fit_transform(train_labels)
        val_labels_binary = mlb.transform(val_labels)
        test_labels_binary = mlb.transform(test_labels)
        
        # Initialize the Logistic Regression classifier with OneVsRest strategy
        clf = OneVsRestClassifier(LogisticRegression(max_iter=20000, class_weight='balanced', C=0.2, penalty='l2', solver='lbfgs', random_state=1234))

        # Timing the training process
        start_time = time.time()
        clf.fit(train_features, train_labels_binary)
        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.2f} seconds.\n")
        f.write(f"Training completed in {train_time:.2f} seconds.\n")
        
        # Timing the validation process
        start_time = time.time()
        val_predictions = clf.predict(val_features)
        val_time = time.time() - start_time
        print(f"Validation predictions completed in {val_time:.2f} seconds.")
        f.write(f"Validation predictions completed in {val_time:.2f} seconds.\n")
        
        # Evaluate on the validation data
        val_micro_f1 = f1_score(val_labels_binary, val_predictions, average='micro')
        val_macro_f1 = f1_score(val_labels_binary, val_predictions, average='macro')
        val_report = classification_report(val_labels_binary, val_predictions, zero_division=0)

        print("\nValidation Set Performance:")
        print(f"F1 Score (micro): {val_micro_f1}")
        print(f"F1 Score (macro): {val_macro_f1}")
        print("\nClassification Report (Validation):\n\n", val_report)
        f.write("\nValidation Set Performance:\n")
        f.write(f"F1 Score (micro): {val_micro_f1}\n")
        f.write(f"F1 Score (macro): {val_macro_f1}\n")
        f.write("\nClassification Report (Validation):\n\n" + val_report + "\n")
        
        # Timing the test process
        start_time = time.time()
        test_predictions = clf.predict(test_features)
        test_time = time.time() - start_time
        print(f"Test predictions completed in {test_time:.2f} seconds.")
        f.write(f"Test predictions completed in {test_time:.2f} seconds.\n")
        
        # Evaluate on the test data
        test_micro_f1 = f1_score(test_labels_binary, test_predictions, average='micro')
        test_macro_f1 = f1_score(test_labels_binary, test_predictions, average='macro')
        test_report = classification_report(test_labels_binary, test_predictions, zero_division=0)

        print("\nTest Set Performance:")
        print(f"F1 Score (micro): {test_micro_f1}")
        print(f"F1 Score (macro): {test_macro_f1}")
        print("\nClassification Report (Test):\n\n", test_report)
        f.write("\nTest Set Performance:\n")
        f.write(f"F1 Score (micro): {test_micro_f1}\n")
        f.write(f"F1 Score (macro): {test_macro_f1}\n")
        f.write("\nClassification Report (Test):\n\n" + test_report)
