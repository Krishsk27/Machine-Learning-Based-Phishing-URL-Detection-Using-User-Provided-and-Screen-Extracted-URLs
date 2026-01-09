# src/evaluate_models.py
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import feature extractor
from feature_extractor import get_url_features, feature_names

# --- CONFIGURATION ---
current_file_path = os.path.abspath(__file__)
src_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_directory)

URL_DATASET_PATH = os.path.join(project_root, 'dataset', 'phishing_urls.csv')
SMS_DATASET_PATH = os.path.join(project_root, 'dataset', 'spam.csv')
MODELS_DIR = os.path.join(project_root, 'models')
ACCURACY_DIR = os.path.join(project_root, 'accuracy')

# Create 'accuracy' folder if it doesn't exist
if not os.path.exists(ACCURACY_DIR):
    os.makedirs(ACCURACY_DIR)
    print(f"Created folder: {ACCURACY_DIR}")

def save_report_to_file(filename, report_text):
    filepath = os.path.join(ACCURACY_DIR, filename)
    with open(filepath, "w") as f:
        f.write(report_text)
    print(f"📄 Report saved to: {filepath}")

def evaluate_url_model():
    print("\n" + "="*50)
    print("📊 EVALUATING URL PHISHING MODEL (XGBoost)")
    print("="*50)

    # 1. Load Model
    model_path = os.path.join(MODELS_DIR, "url_classifier.json")
    if not os.path.exists(model_path):
        print("❌ Model not found. Train it first!")
        return

    print("Loading model...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # 2. Load Data
    if not os.path.exists(URL_DATASET_PATH):
        print(f"❌ Dataset not found at {URL_DATASET_PATH}")
        return
    
    print("Loading dataset...")
    df = pd.read_csv(URL_DATASET_PATH)
    df.columns = ['url', 'label']
    df['label'] = df['label'].map({'bad': 1, 'good': 0})
    
    # Test Sample (Increase n=5000 for better accuracy check)
    print("Extracting features from test sample (2000 rows)...")
    df_test = df.sample(n=2000, random_state=42)

    features_list = []
    labels_list = []
    
    for index, row in df_test.iterrows():
        try:
            feats = get_url_features(str(row['url']))
            if feats:
                features_list.append(feats)
                labels_list.append(row['label'])
        except:
            continue

    X = pd.DataFrame(features_list, columns=feature_names)
    y_true = np.array(labels_list)

    # 3. Predict
    print("Running predictions...")
    y_pred = model.predict(X)

    # 4. Generate Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Safe', 'Phishing'])
    cm = confusion_matrix(y_true, y_pred)

    # 5. Print & Save Text Report
    output_text = f"URL MODEL EVALUATION\n{'='*30}\n"
    output_text += f"Accuracy: {acc * 100:.2f}%\n\n"
    output_text += "Classification Report:\n"
    output_text += report
    
    print(output_text)
    save_report_to_file("url_model_report.txt", output_text)

    # 6. Save Confusion Matrix Graph
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Safe', 'Phishing'], 
                    yticklabels=['Safe', 'Phishing'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'URL Model Confusion Matrix (Acc: {acc*100:.1f}%)')
        
        # Save Plot
        plot_path = os.path.join(ACCURACY_DIR, "url_confusion_matrix.png")
        plt.savefig(plot_path)
        print(f"📈 Graph saved to: {plot_path}")
        # plt.show() # Uncomment to see popup window
        plt.close()
    except Exception as e:
        print("Could not save graph:", e)

def evaluate_nlp_model():
    print("\n" + "="*50)
    print("📊 EVALUATING SMS/TEXT MODEL (NLP)")
    print("="*50)

    # 1. Load Models
    model_path = os.path.join(MODELS_DIR, "nlp_model.pkl")
    vect_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    
    if not os.path.exists(model_path):
        print("❌ NLP Model not found.")
        return

    with open(model_path, "rb") as f:
        nlp_model = pickle.load(f)
    with open(vect_path, "rb") as f:
        vectorizer = pickle.load(f)

    # 2. Load Data
    if not os.path.exists(SMS_DATASET_PATH):
        print(f"❌ Dataset not found at {SMS_DATASET_PATH}")
        return

    print("Loading dataset...")
    df = pd.read_csv(SMS_DATASET_PATH, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']

    X = vectorizer.transform(df['message'])
    y_true = df['label']

    # 3. Predict
    y_pred = nlp_model.predict(X)

    # 4. Generate Metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # 5. Print & Save Text Report
    output_text = f"NLP SMS MODEL EVALUATION\n{'='*30}\n"
    output_text += f"Accuracy: {acc * 100:.2f}%\n\n"
    output_text += "Classification Report:\n"
    output_text += report
    
    print(output_text)
    save_report_to_file("nlp_model_report.txt", output_text)

    # 6. Save Graph
    try:
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=['Ham', 'Spam'], 
                    yticklabels=['Ham', 'Spam'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'SMS Model Confusion Matrix (Acc: {acc*100:.1f}%)')
        
        plot_path = os.path.join(ACCURACY_DIR, "nlp_confusion_matrix.png")
        plt.savefig(plot_path)
        print(f"📈 Graph saved to: {plot_path}")
        plt.close()
    except Exception as e:
        print("Could not save graph:", e)

if __name__ == "__main__":
    evaluate_url_model()
    evaluate_nlp_model()
    print(f"\n✅ All reports saved in: {ACCURACY_DIR}")