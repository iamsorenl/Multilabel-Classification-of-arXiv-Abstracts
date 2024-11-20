import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


def stacked_classifier_model(train, val, test, outfile):
    with open(outfile, 'w') as f:
        # Combine titles and summaries
        train_text = train['titles'] + " " + train['summaries']
        val_text = val['titles'] + " " + val['summaries']
        test_text = test['titles'] + " " + test['summaries']

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=30000, stop_words='english', ngram_range=(1, 2), min_df=2)
        train_features = vectorizer.fit_transform(train_text)
        val_features = vectorizer.transform(val_text)
        test_features = vectorizer.transform(test_text)

        # Scale features
        scaler = StandardScaler(with_mean=False)
        train_features = scaler.fit_transform(train_features)
        val_features = scaler.transform(val_features)
        test_features = scaler.transform(test_features)

        # MultiLabelBinarizer for labels
        mlb = MultiLabelBinarizer()
        train_labels_binary = mlb.fit_transform(train['terms'])
        val_labels_binary = mlb.transform(val['terms'])
        test_labels_binary = mlb.transform(test['terms'])

        # Define base estimators
        base_estimators = [
            ('rf', MultiOutputClassifier(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1))),
            ('ridge', MultiOutputClassifier(RidgeClassifier()))
        ]

        # Add Gradient Boosting Classifier as a separate model or in the final estimator
        final_estimator = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=50, random_state=42))

        # Define stacking classifier
        clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=final_estimator,
            n_jobs=-1
        )

        # Train the stacking model
        start_time = time.time()
        clf.fit(train_features, train_labels_binary)
        train_time = time.time() - start_time
        print(f"\nTraining completed in {train_time:.2f} seconds.\n")
        f.write(f"Training completed in {train_time:.2f} seconds.\n")

        # Make predictions on the validation set
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
