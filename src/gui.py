# src/gui_main.py
import sys
import os
import time
import re
import cv2
import mss
import numpy as np
import pytesseract
import xgboost as xgb
import pickle
import pandas as pd
import shap
import matplotlib.pyplot as plt

# FIX 1: Use 'backend_qtagg' which automatically detects PyQt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                            QLineEdit, QTabWidget, QMessageBox, QProgressBar)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QFont, QColor

# Import feature extractor
from feature_extractor import get_url_features, feature_names

# FIX 2: ROBUST PATH CONFIGURATION (Works from anywhere)
current_file_path = os.path.abspath(__file__)          # D:\...\src\gui_main.py
src_directory = os.path.dirname(current_file_path)     # D:\...\src
project_root = os.path.dirname(src_directory)          # D:\... (Root)
MODELS_DIR = os.path.join(project_root, 'models')      # D:\...\models

# Check Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- WORKER THREAD FOR SCREEN MONITORING ---
class MonitorThread(QThread):
    alert_signal = pyqtSignal(str, str)

    def __init__(self):
        super().__init__()
        self.running = False
        
        # Load Models using Absolute Paths
        print(f"Loading models from: {MODELS_DIR}")
        self.url_model = xgb.XGBClassifier()
        self.url_model.load_model(os.path.join(MODELS_DIR, "url_classifier.json"))
        
        with open(os.path.join(MODELS_DIR, "nlp_model.pkl"), "rb") as f:
            self.nlp_model = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "vectorizer.pkl"), "rb") as f:
            self.vectorizer = pickle.load(f)

    def run(self):
        self.running = True
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            prev_gray = None
            
            while self.running:
                # 1. Capture & Diff Check
                img = np.array(sct.grab(monitor))
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                small = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)

                if prev_gray is not None:
                    err = np.sum((small.astype("float") - prev_gray.astype("float")) ** 2)
                    err /= float(small.shape[0] * small.shape[1])
                    if err < 500: 
                        time.sleep(1)
                        continue
                prev_gray = small

                # 2. OCR
                try:
                    text = pytesseract.image_to_string(gray, config='--psm 11')
                except Exception as e:
                    print(f"OCR Error: {e}")
                    text = ""
                
                # 3. Analyze
                if len(text) > 10:
                    self.check_content(text)
                
                time.sleep(2)

    def check_content(self, text):
        # Check URLs
        urls = re.findall(r'(https?://[^\s]+|www\.[^\s]+)', text)
        for url in urls:
            if len(url) < 5: continue
            feats = get_url_features(url)
            if feats:
                df = pd.DataFrame([feats], columns=feature_names)
                pred = self.url_model.predict(df)[0]
                if pred == 1:
                    self.alert_signal.emit("PHISHING LINK", url)

        # Check Text
        lines = text.split('\n')
        for line in lines:
            if len(line.split()) > 4:
                vect = self.vectorizer.transform([line])
                pred = self.nlp_model.predict(vect)[0]
                if pred == 'spam':
                    if any(x in line.lower() for x in ['urgent', 'verify', 'bank', 'winner']):
                        self.alert_signal.emit("SCAM MESSAGE", line)

    def stop(self):
        self.running = False
        self.wait()

# --- MAIN GUI APPLICATION ---
class PhishingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Phishing Detector & Explainer")
        self.setGeometry(100, 100, 900, 700)
        self.setStyleSheet("background-color: #f0f0f0; font-size: 14px;")

        self.monitor_thread = MonitorThread()
        self.monitor_thread.alert_signal.connect(self.show_alert)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        self.tab_dashboard = QWidget()
        self.setup_dashboard()
        tabs.addTab(self.tab_dashboard, "🛡️ Live Monitor")

        self.tab_manual = QWidget()
        self.setup_manual_check()
        tabs.addTab(self.tab_manual, "🔍 Manual Analysis")
        
        # Load Model for Manual Check
        self.manual_model = xgb.XGBClassifier()
        self.manual_model.load_model(os.path.join(MODELS_DIR, "url_classifier.json"))

    def setup_dashboard(self):
        layout = QVBoxLayout()
        
        self.status_label = QLabel("Status: SYSTEM IDLE")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("font-size: 20px; font-weight: bold; color: #555;")
        layout.addWidget(self.status_label)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Monitoring")
        self.btn_start.setStyleSheet("background-color: #28a745; color: white; padding: 15px;")
        self.btn_start.clicked.connect(self.start_monitoring)
        
        self.btn_stop = QPushButton("Stop Monitoring")
        self.btn_stop.setStyleSheet("background-color: #dc3545; color: white; padding: 15px;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_monitoring)

        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Detection logs will appear here...")
        layout.addWidget(self.log_box)
        
        self.tab_dashboard.setLayout(layout)

    def setup_manual_check(self):
        layout = QVBoxLayout()
        
        input_label = QLabel("Enter Suspicious URL:")
        layout.addWidget(input_label)
        
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("http://example.com")
        layout.addWidget(self.url_input)
        
        self.btn_analyze = QPushButton("Analyze URL & Explain Decision")
        self.btn_analyze.setStyleSheet("background-color: #007bff; color: white; padding: 10px;")
        self.btn_analyze.clicked.connect(self.analyze_manual_url)
        layout.addWidget(self.btn_analyze)

        # Area for SHAP Plot
        self.figure = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.result_label = QLabel("")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.result_label)

        self.tab_manual.setLayout(layout)

    def start_monitoring(self):
        self.monitor_thread.start()
        self.status_label.setText("Status: MONITORING ACTIVE")
        self.status_label.setStyleSheet("color: green; font-weight: bold; font-size: 20px;")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_monitoring(self):
        self.monitor_thread.stop()
        self.status_label.setText("Status: SYSTEM PAUSED")
        self.status_label.setStyleSheet("color: red; font-weight: bold; font-size: 20px;")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def show_alert(self, alert_type, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] ⚠️ {alert_type}: {message}")

    def analyze_manual_url(self):
        url = self.url_input.text()
        if not url: return

        feats = get_url_features(url)
        if not feats:
            self.result_label.setText("Invalid URL Format")
            return

        df = pd.DataFrame([feats], columns=feature_names)
        
        prob = self.manual_model.predict_proba(df)[0][1] 
        is_phishing = prob > 0.5
        
        color = "red" if is_phishing else "green"
        text = "PHISHING DETECTED" if is_phishing else "SAFE URL"
        self.result_label.setText(f"Result: {text} (Risk Score: {prob*100:.1f}%)")
        self.result_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 18px;")

        self.figure.clear()
        try:
            explainer = shap.TreeExplainer(self.manual_model)
            shap_values = explainer.shap_values(df)

            ax = self.figure.add_subplot(111)
            feature_importance = pd.Series(shap_values[0], index=feature_names)
            feature_importance.nlargest(5).plot(kind='barh', ax=ax, color=color)
            ax.set_title("Why did the AI make this decision?")
            ax.set_xlabel("Impact on Risk Score")
            self.figure.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"SHAP Error: {e}")