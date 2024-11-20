import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

def preprocess_labels(data):
    # Split labels into top-level and sub-level, handling cases without a dot
    data['top_level'] = data['terms'].apply(lambda x: [label.split('.')[0] for label in x])
    data['sub_level'] = data['terms'].apply(lambda x: [label.split('.')[1] if '.' in label else 'unknown' for label in x])
    return data

def hierarchical_model(train, val, test, outfile):
    with open(outfile, 'w') as f:
        # Preprocess labels
        train = preprocess_labels(train)
        val = preprocess_labels(val)
        test = preprocess_labels(test)

        # Combine titles and summaries into text features
        train_text = train['titles'] + " " + train['summaries']
        val_text = val['titles'] + " " + val['summaries']
        test_text = test['titles'] + " " + test['summaries']

        # Vectorize text data with TF-IDF
        vectorizer = TfidfVectorizer(max_features=30000, stop_words='english', ngram_range=(1, 2), min_df=2)
        train_features = vectorizer.fit_transform(train_text)
        val_features = vectorizer.transform(val_text)
        test_features = vectorizer.transform(test_text)

        # MultiLabelBinarizer for top-level categories
        top_level_mlb = MultiLabelBinarizer()
        train_top_level = top_level_mlb.fit_transform(train['top_level'])
        val_top_level = top_level_mlb.transform(val['top_level'])
        test_top_level = top_level_mlb.transform(test['top_level'])

        # Train top-level classifier
        top_level_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        start_time = time.time()
        top_level_clf.fit(train_features, train_top_level)
        train_top_time = time.time() - start_time
        print(f"\nTop-level training completed in {train_top_time:.2f} seconds.\n")
        f.write(f"Top-level training completed in {train_top_time:.2f} seconds.\n")

        # Predict top-level categories
        val_top_level_pred = top_level_clf.predict(val_features)
        test_top_level_pred = top_level_clf.predict(test_features)

        # Evaluate top-level classifier
        val_top_micro_f1 = f1_score(val_top_level, val_top_level_pred, average='micro')
        val_top_macro_f1 = f1_score(val_top_level, val_top_level_pred, average='macro')
        val_top_report = classification_report(val_top_level, val_top_level_pred, zero_division=1)

        print(f"\nTop-Level Validation Micro F1: {val_top_micro_f1}, Macro F1: {val_top_macro_f1}")
        f.write(f"\nTop-Level Validation Micro F1: {val_top_micro_f1}, Macro F1: {val_top_macro_f1}\n")
        print("\nTop-Level Validation Classification Report:")
        print(val_top_report)
        f.write("\nTop-Level Validation Classification Report:\n" + val_top_report)

        test_top_micro_f1 = f1_score(test_top_level, test_top_level_pred, average='micro')
        test_top_macro_f1 = f1_score(test_top_level, test_top_level_pred, average='macro')
        test_top_report = classification_report(test_top_level, test_top_level_pred, zero_division=1)

        print(f"\nTop-Level Test Micro F1: {test_top_micro_f1}, Macro F1: {test_top_macro_f1}")
        f.write(f"\nTop-Level Test Micro F1: {test_top_micro_f1}, Macro F1: {test_top_macro_f1}\n")
        print("\nTop-Level Test Classification Report:")
        print(test_top_report)
        f.write("\nTop-Level Test Classification Report:\n" + test_top_report + "\n")

        # MultiLabelBinarizer for subcategories
        sub_level_mlb = MultiLabelBinarizer()
        train_sub_level = sub_level_mlb.fit_transform(train['sub_level'])
        val_sub_level = sub_level_mlb.transform(val['sub_level'])
        test_sub_level = sub_level_mlb.transform(test['sub_level'])

        # Train fine-grained classifiers for each top-level category
        sub_level_clfs = {}
        for category in top_level_mlb.classes_:
            indices = [i for i, labels in enumerate(train['top_level']) if category in labels]
            sub_level_clfs[category] = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            sub_features = train_features[indices]
            sub_labels = train_sub_level[indices]
            sub_level_clfs[category].fit(sub_features, sub_labels)

        # Predict subcategories
        val_predictions = np.zeros(val_sub_level.shape)
        for i, category in enumerate(top_level_mlb.classes_):
            indices = [idx for idx, label in enumerate(val_top_level_pred[:, i]) if label == 1]
            if len(indices) > 0:
                sub_clf = sub_level_clfs[category]
                sub_preds = sub_clf.predict(val_features[indices])
                val_predictions[indices, :] = sub_preds

        test_predictions = np.zeros(test_sub_level.shape)
        for i, category in enumerate(top_level_mlb.classes_):
            indices = [idx for idx, label in enumerate(test_top_level_pred[:, i]) if label == 1]
            if len(indices) > 0:
                sub_clf = sub_level_clfs[category]
                sub_preds = sub_clf.predict(test_features[indices])
                test_predictions[indices, :] = sub_preds

        # Evaluate sub-level classifier
        val_micro_f1 = f1_score(val_sub_level, val_predictions, average='micro', zero_division=1)
        val_macro_f1 = f1_score(val_sub_level, val_predictions, average='macro', zero_division=1)
        val_report = classification_report(val_sub_level, val_predictions, zero_division=1)

        print(f"\nValidation Micro F1: {val_micro_f1}, Macro F1: {val_macro_f1}")
        f.write(f"\nValidation Micro F1: {val_micro_f1}, Macro F1: {val_macro_f1}\n")
        print("\nValidation Sub-Level Classification Report:")
        print(val_report)
        f.write("\nValidation Sub-Level Classification Report:\n" + val_report + "\n")

        test_micro_f1 = f1_score(test_sub_level, test_predictions, average='micro', zero_division=1)
        test_macro_f1 = f1_score(test_sub_level, test_predictions, average='macro', zero_division=1)
        test_report = classification_report(test_sub_level, test_predictions, zero_division=1)

        print(f"\nTest Micro F1: {test_micro_f1}, Macro F1: {test_macro_f1}")
        f.write(f"\nTest Micro F1: {test_micro_f1}, Macro F1: {test_macro_f1}\n")
        print("\nTest Sub-Level Classification Report:")
        print(test_report)
        f.write("\nTest Sub-Level Classification Report:\n" + test_report)
