import time
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

def svm_model(train, val, test, outfile):
    with open(outfile, 'w') as f:
        # Combine titles and summaries into text features
        train_text = train['titles'] + " " + train['summaries']
        val_text = val['titles'] + " " + val['summaries']
        test_text = test['titles'] + " " + test['summaries']

        # Vectorize text data with TF-IDF
        vectorizer = TfidfVectorizer(max_features=30000, stop_words='english', ngram_range=(1, 2), min_df=2)
        train_features = vectorizer.fit_transform(train_text)
        val_features = vectorizer.transform(val_text)
        test_features = vectorizer.transform(test_text)

        # Scale features for LinearSVC
        scaler = StandardScaler(with_mean=False)
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        # Transform labels into binary format
        mlb = MultiLabelBinarizer()
        train_labels_binary = mlb.fit_transform(train['terms'])
        val_labels_binary = mlb.transform(val['terms'])
        test_labels_binary = mlb.transform(test['terms'])

        # Define the LinearSVC pipeline
        clf = LinearSVC(C=10, max_iter=5000, random_state=42)

        # Train the model
        start_time = time.time()
        clf.fit(train_features, train_labels_binary)
        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.2f} seconds.\n")
        f.write(f"Training completed in {train_time:.2f} seconds.\n")

        # Make predictions on validation set
        start_time = time.time()
        val_predictions = clf.predict(val_features)
        val_time = time.time() - start_time
        print(f"Validation predictions completed in {val_time:.2f} seconds.")
        f.write(f"Validation predictions completed in {val_time:.2f} seconds.\n")

        # Evaluate on validation set
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

        # Make predictions on the test set
        start_time = time.time()
        test_predictions = clf.predict(test_features)
        test_time = time.time() - start_time
        print(f"Test predictions completed in {test_time:.2f} seconds.")
        f.write(f"Test predictions completed in {test_time:.2f} seconds.\n")

        # Evaluate on test set
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
        f.write("\nClassification Report (Test):\n\n" + test_report + "\n")
