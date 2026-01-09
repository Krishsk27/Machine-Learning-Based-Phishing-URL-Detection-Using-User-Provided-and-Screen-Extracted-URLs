# src/train_models.py
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import our custom extractor
from feature_extractor import get_url_features, feature_names

# --- CONFIGURATION ---
URL_DATASET_PATH = '../dataset/phishing_urls.csv'
SMS_DATASET_PATH = '../dataset/spam.csv'
MODELS_DIR = '../models'

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def train_url_model():
    print("\n[1/2] --- Starting URL Model Training ---")
    
    if not os.path.exists(URL_DATASET_PATH):
        print(f"Error: {URL_DATASET_PATH} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(URL_DATASET_PATH)
    df.columns = ['url', 'label']
    
    # Map Labels
    df['label'] = df['label'].map({'bad': 1, 'good': 0})
    
    # Sample for speed
    print("Extracting features from 15,000 URLs (Sampling for speed)...")
    df_sample = df.sample(n=15000, random_state=42)

    # 2. Extract Features
    features_list = []
    labels_list = []
    
    for index, row in df_sample.iterrows():
        try:
            feats = get_url_features(str(row['url']))
            if feats:
                features_list.append(feats)
                labels_list.append(row['label'])
        except:
            continue

    X = pd.DataFrame(features_list, columns=feature_names)
    y = np.array(labels_list)

    # 3. Train XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # FIX APPLIED HERE: Removed 'use_label_encoder'
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 4. Save Model
    acc = accuracy_score(y_test, model.predict(X_test))
    
    # FIX APPLIED HERE: Use get_booster() to save safely
    model.get_booster().save_model(f"{MODELS_DIR}/url_classifier.json")
    
    print(f"URL Model Saved! Accuracy: {acc*100:.2f}%")

def train_nlp_model():
    print("\n[2/2] --- Starting NLP (Text) Model Training ---")
    
    if not os.path.exists(SMS_DATASET_PATH):
        print(f"Error: {SMS_DATASET_PATH} not found.")
        return

    # 1. Load Data
    df = pd.read_csv(SMS_DATASET_PATH, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    # 2. Vectorize
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['message'])
    y = df['label']

    # 3. Train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nlp_model = RandomForestClassifier(n_estimators=100, random_state=42)
    nlp_model.fit(X_train, y_train)

    # 4. Save
    acc = accuracy_score(y_test, nlp_model.predict(X_test))
    
    with open(f"{MODELS_DIR}/nlp_model.pkl", "wb") as f:
        pickle.dump(nlp_model, f)
    
    with open(f"{MODELS_DIR}/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
        
    print(f"NLP Model & Vectorizer Saved! Accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    train_url_model()
    train_nlp_model()
    print("\n--- All Systems Ready ---")