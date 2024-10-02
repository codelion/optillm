import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def preprocess_data(data):
    X = []
    y = []
    for item in data:
        prompt = item['prompt']
        results = item['results']
        try:
            best_approach = min(results, key=lambda x: x.get('rank', float('inf')))['approach']
            X.append(prompt)
            y.append(best_approach)
        except (KeyError, ValueError) as e:
            logger.warning(f"Error encountered: {e}. Skipping this item.")
            logger.debug(f"Problematic item: {item}")
    return X, y

def extract_features(X):
    vectorizer = TfidfVectorizer(max_features=1000)
    X_features = vectorizer.fit_transform(X)
    return X_features, vectorizer

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=1))

def select_approach(model, vectorizer, prompt, effort, approaches, token_usage):
    X_input = vectorizer.transform([prompt])
    probabilities = model.predict_proba(X_input)[0]
    
    sorted_approaches = sorted(zip(approaches, probabilities), key=lambda x: x[1], reverse=True)
    
    if effort == 1:
        return sorted_approaches[0][0]
    elif effort == 0:
        return min(token_usage, key=token_usage.get)
    else:
        scores = []
        for approach, prob in sorted_approaches:
            if approach in token_usage:
                normalized_tokens = (token_usage[approach] - min(token_usage.values())) / (max(token_usage.values()) - min(token_usage.values()))
                score = effort * prob + (1 - effort) * (1 - normalized_tokens)
                scores.append((approach, score))
        return max(scores, key=lambda x: x[1])[0] if scores else sorted_approaches[0][0]

def main():
    data = load_data('optillm_dataset_1.jsonl')
    X, y = preprocess_data(data)
    
    if not X or not y:
        logger.error("No valid data after preprocessing. Check your dataset.")
        return

    X_features, vectorizer = extract_features(X)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_features, y_encoded, test_size=0.2, random_state=42)
    
    model = train_model(X_train, y_train)
    
    evaluate_model(model, X_test, y_test)
    
    joblib.dump(model, 'optillm_approach_classifier.joblib')
    joblib.dump(vectorizer, 'optillm_vectorizer.joblib')
    joblib.dump(label_encoder, 'optillm_label_encoder.joblib')
    
    token_usage = {approach: [] for approach in set(y)}
    for item in data:
        for result in item['results']:
            approach = result.get('approach')
            tokens = result.get('tokens')
            if approach and tokens is not None:
                if approach not in token_usage:
                    token_usage[approach] = []
                token_usage[approach].append(tokens)
    
    avg_token_usage = {approach: np.mean(usage) if usage else 0 for approach, usage in token_usage.items()}
    
    prompt = "Write a Python function to calculate the Fibonacci sequence."
    effort = 0.1
    approaches = label_encoder.classes_
    
    selected_approach = select_approach(model, vectorizer, prompt, effort, approaches, avg_token_usage)
    print(f"Selected approach for prompt '{prompt}' with effort {effort}: {selected_approach}")

if __name__ == "__main__":
    main()