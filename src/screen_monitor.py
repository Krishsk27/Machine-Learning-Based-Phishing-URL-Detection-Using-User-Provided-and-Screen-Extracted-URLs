# src/screen_monitor.py
import time
import re
import cv2
import mss
import numpy as np
import pytesseract
import xgboost as xgb
import pickle
import threading
from feature_extractor import get_url_features, feature_names

# --- CONFIGURATION ---
# Set your Tesseract path (Windows Example):
# If on Mac/Linux, you can usually comment this line out.
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

MODELS_DIR = '../models'

class PhishingMonitor:
    def __init__(self):
        print("Initializing Phishing Monitor...")
        self.running = False
        
        # 1. Load URL Model
        self.url_model = xgb.XGBClassifier()
        self.url_model.load_model(f"{MODELS_DIR}/url_classifier.json")
        
        # 2. Load NLP Models
        with open(f"{MODELS_DIR}/nlp_model.pkl", "rb") as f:
            self.nlp_model = pickle.load(f)
        with open(f"{MODELS_DIR}/vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)

        print("Models Loaded Successfully.")

    def start_monitoring(self):
        self.running = True
        # Run the loop in a separate thread
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True # Kills thread if main app closes
        monitor_thread.start()
        print("Background Monitoring Started... (Press Ctrl+C to stop)")
        
        # Keep main thread alive
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            self.stop_monitoring()

    def stop_monitoring(self):
        self.running = False
        print("Stopping Monitor...")

    def _monitor_loop(self):
        with mss.mss() as sct:
            # Monitor full screen
            monitor = sct.monitors[1] 
            prev_gray = None

            while self.running:
                # 1. Capture Screen
                screenshot = np.array(sct.grab(monitor))
                
                # 2. Preprocessing (Gray + Resize for speed)
                gray = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2GRAY)
                # Resize to 50% to speed up "Diff" check (not OCR)
                small_gray = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)

                # 3. Check for Changes (Optimization)
                if prev_gray is not None:
                    # Calculate Mean Squared Error (Difference between frames)
                    err = np.sum((small_gray.astype("float") - prev_gray.astype("float")) ** 2)
                    err /= float(small_gray.shape[0] * small_gray.shape[1])
                    
                    if err < 500: # Threshold: If screen hasn't changed much, skip OCR
                        time.sleep(1) # Wait longer if user is idle
                        continue
                
                prev_gray = small_gray

                # 4. Perform OCR (On the full resolution image)
                # psm 11 = Sparse text (good for reading websites/messages scattered on screen)
                text_content = pytesseract.image_to_string(gray, config='--psm 11')

                # 5. Analyze Content
                if len(text_content.strip()) > 5:
                    self._analyze_text(text_content)
                
                # Wait before next scan to save CPU
                time.sleep(2)

    def _analyze_text(self, text):
        # --- A. URL Detection ---
        # Find anything looking like a domain or link
        urls = re.findall(r'(https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.(?:com|net|org|co|in))', text)
        
        for url in urls:
            # Filter out noise (e.g., 'Google.com' is safe, ignore simple text)
            if '.' not in url or len(url) < 5: continue
            
            # Extract features
            feats = get_url_features(url)
            if not feats: continue
            
            # Predict (Dataframe required for XGBoost feature names alignment)
            import pandas as pd
            feat_df = pd.DataFrame([feats], columns=feature_names)
            
            pred = self.url_model.predict(feat_df)[0]
            
            if pred == 1:
                print(f" [!!!] PHISHING ALERT: Suspicious URL detected -> {url}")

        # --- B. NLP Message Detection ---
        # Split into lines to analyze sentences
        lines = text.split('\n')
        for line in lines:
            if len(line.split()) > 4: # Only analyze sentences with >4 words
                vect = self.vectorizer.transform([line])
                pred = self.nlp_model.predict(vect)[0]
                
                if pred == 'spam':
                    # Double check confidence or keywords to reduce false positives
                    keywords = ['urgent', 'winner', 'click', 'account', 'verify', 'bank']
                    if any(k in line.lower() for k in keywords):
                        print(f" [!!!] SCAM TEXT ALERT: Suspicious message -> '{line.strip()}'")

# Entry Point
if __name__ == "__main__":
    app = PhishingMonitor()
    app.start_monitoring()