import time
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier
from scipy.sparse import hstack

def knn_model(train, val, test, outfile, max_features=30000, n_neighbors=5):
    # Open the output file for writing
    with open(outfile, 'w') as f:
        # Vectorize separately using TF-IDF for titles and summaries
        title_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2), min_df=2)
        summary_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2), min_df=2)

        # Fit and transform on training data; transform on validation/test data
        train_title_features = title_vectorizer.fit_transform(train['titles'])
        val_title_features = title_vectorizer.transform(val['titles'])
        test_title_features = title_vectorizer.transform(test['titles'])

        train_summary_features = summary_vectorizer.fit_transform(train['summaries'])
        val_summary_features = summary_vectorizer.transform(val['summaries'])
        test_summary_features = summary_vectorizer.transform(test['summaries'])

        # Combine title and summary features using scipy.sparse.hstack for efficient sparse matrix operations
        train_features = hstack([train_title_features, train_summary_features])
        val_features = hstack([val_title_features, val_summary_features])
        test_features = hstack([test_title_features, test_summary_features])

        # Optionally scale features (important for distance-based models like k-NN)
        scaler = StandardScaler(with_mean=False)
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        # Transform labels using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        train_labels_binary = mlb.fit_transform(train['terms'])
        val_labels_binary = mlb.transform(val['terms'])
        test_labels_binary = mlb.transform(test['terms'])

        # Define and fit the MultiOutputClassifier with KNeighborsClassifier as the base estimator
        clf = MultiOutputClassifier(KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1))
        
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

        # Make predictions on the test set
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
        f.write("\nClassification Report (Test):\n\n" + test_report + "\n")
