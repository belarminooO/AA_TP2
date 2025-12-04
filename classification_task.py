import pickle
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from nltk.stem import PorterStemmer

# 1. Load Data
print("Loading data...")
fn = "imdbFull.p"
try:
    with open(fn, 'rb') as f:
        D = pickle.load(f)
    Docs = D.data
    y = D.target
    print(f"Loaded {len(Docs)} documents.")
except FileNotFoundError:
    print(f"Error: {fn} not found. Please ensure the file is in the correct directory.")
    exit()

# 2. Preprocessing Function
def clean_text(text):
    # Remove HTML tags (replace with space)
    text = text.replace('<br />', ' ')
    # Keep only letters and latin characters
    text = re.sub(r'[^a-zA-Z\u00C0-\u00FF]+', ' ', text)
    return text

print("Cleaning documents...")
Docs_clean = [clean_text(doc) for doc in Docs]

# 3. Stemming Function
def stem_documents(docs):
    stemmer = PorterStemmer()
    # Simple stemming: split by space, stem, join back
    # Note: This might be slow for large datasets
    return [' '.join([stemmer.stem(w) for w in d.split()]) for d in docs]

# 4. Classification Helper Function
def run_classification(documents, labels, use_stemming=False):
    print(f"\n--- Running Classification (Stemming: {use_stemming}) ---")
    
    # Apply stemming if requested
    if use_stemming:
        print("Applying stemming (this may take a while)...")
        docs_to_use = stem_documents(documents)
    else:
        docs_to_use = documents

    # Split Data (Train/Test)
    # Using a fixed random_state for reproducibility
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        docs_to_use, labels, test_size=0.25, random_state=42
    )
    print(f"Split data: {len(X_train_raw)} train, {len(X_test_raw)} test.")

    # Vectorization
    print("Vectorizing...")
    # Using parameters from your notebook/example
    tfidf = TfidfVectorizer(min_df=5, token_pattern=r'\b\w\w\w\w+\b')
    
    # Fit on TRAIN, transform TRAIN and TEST
    X_train = tfidf.fit_transform(X_train_raw)
    X_test = tfidf.transform(X_test_raw)
    
    print(f"Vocabulary size: {len(tfidf.get_feature_names_out())}")

    # Classification (Logistic Regression)
    print("Training Logistic Regression...")
    clf = LogisticRegression(penalty='l2', max_iter=1000, C=1, tol=1e-3)
    clf.fit(X_train, y_train)

    # Evaluation
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)

    print(f"Results (Stemming: {use_stemming}):")
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy:  {test_score:.4f}")
    
    return train_score, test_score

# --- Execution ---

# Scenario 1: Without Stemmer
score_train_no_stem, score_test_no_stem = run_classification(Docs_clean, y, use_stemming=False)

# Scenario 2: With Stemmer
score_train_stem, score_test_stem = run_classification(Docs_clean, y, use_stemming=True)

# Comparison
print("\n=== Final Comparison ===")
print(f"No Stemmer - Train: {score_train_no_stem:.4f}, Test: {score_test_no_stem:.4f}")
print(f"With Stemmer - Train: {score_train_stem:.4f}, Test: {score_test_stem:.4f}")
