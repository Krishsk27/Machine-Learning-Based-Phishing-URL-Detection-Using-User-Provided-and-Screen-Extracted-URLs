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
import shap # Still needed for logic, but we won't plot it

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QTextEdit, 
                             QLineEdit, QStackedWidget, QFrame, QGraphicsDropShadowEffect)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QPoint
from PyQt6.QtGui import QFont, QColor, QCursor

from feature_extractor import get_url_features, feature_names
from notifications import DynamicIsland

# --- PATHS ---
current_file_path = os.path.abspath(__file__)
src_directory = os.path.dirname(current_file_path)
project_root = os.path.dirname(src_directory)
MODELS_DIR = os.path.join(project_root, 'models')
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- MACOS THEME CSS ---
MACOS_THEME = """
/* Global Window */
QMainWindow {
    background-color: #1e1e1e;
    border-radius: 15px;
}
QWidget {
    font-family: 'Segoe UI', sans-serif;
    color: #e0e0e0;
}
/* Sidebar */
QFrame#Sidebar {
    background-color: #252526;
    border-top-left-radius: 15px;
    border-bottom-left-radius: 15px;
    border-right: 1px solid #333;
}
QPushButton#SidebarBtn {
    background-color: transparent;
    color: #aaaaaa;
    text-align: left;
    padding: 12px 20px;
    border-radius: 8px;
    font-size: 14px;
    font-weight: 500;
}
QPushButton#SidebarBtn:hover {
    background-color: rgba(255, 255, 255, 0.05);
    color: white;
}
QPushButton#SidebarBtn:checked {
    background-color: #007AFF; 
    color: white;
}
/* Content Area */
QFrame#Content {
    background-color: #1e1e1e;
    border-top-right-radius: 15px;
    border-bottom-right-radius: 15px;
}
/* Cards & Inputs */
QFrame#Card {
    background-color: #2d2d2d;
    border-radius: 12px;
    border: 1px solid #3e3e3e;
}
QLineEdit {
    background-color: #333333;
    border: 1px solid #444;
    border-radius: 8px;
    padding: 10px;
    color: white;
}
QPushButton#ActionBtn {
    background-color: #333;
    border: 1px solid #444;
    color: white;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: bold;
}
QPushButton#ActionBtn:hover {
    background-color: #444;
}
QPushButton#ActionBtn:pressed {
    background-color: #007AFF;
    border-color: #007AFF;
}
QTextEdit {
    background-color: #000000;
    border: 1px solid #333;
    border-radius: 8px;
    color: #00FF41; 
    font-family: 'Consolas', monospace;
}
/* Window Controls */
QPushButton#CloseBtn { background-color: #FF5F56; border-radius: 6px; } 
QPushButton#MinBtn { background-color: #FFBD2E; border-radius: 6px; }   
QPushButton#MaxBtn { background-color: #27C93F; border-radius: 6px; }   
"""

# --- THREAD LOGIC ---
class MonitorThread(QThread):
    alert_signal = pyqtSignal(str, str)
    def __init__(self):
        super().__init__()
        self.running = False
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
                try: text = pytesseract.image_to_string(gray, config='--psm 11')
                except: text = ""
                if len(text) > 10: self.check_content(text)
                time.sleep(2)

    def check_content(self, text):
        urls = re.findall(r'(https?://[^\s]+|www\.[^\s]+)', text)
        for url in urls:
            if len(url) < 5: continue
            feats = get_url_features(url)
            if feats:
                df = pd.DataFrame([feats], columns=feature_names)
                pred = self.url_model.predict(df)[0]
                if pred == 1: self.alert_signal.emit("PHISHING LINK", url)
        lines = text.split('\n')
        for line in lines:
            if len(line.split()) > 4:
                vect = self.vectorizer.transform([line])
                pred = self.nlp_model.predict(vect)[0]
                if pred == 'spam':
                    if any(x in line.lower() for x in ['urgent', 'verify', 'bank', 'winner']):
                        self.alert_signal.emit("SCAM MESSAGE", line[:50] + "...")
    def stop(self):
        self.running = False
        self.wait()

# --- MAIN APP ---
class PhishingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.resize(1000, 700)
        
        self.container = QFrame(self)
        self.container.setObjectName("Container")
        self.container.setStyleSheet("#Container { background-color: #1e1e1e; border-radius: 15px; border: 1px solid #333; }")
        self.setCentralWidget(self.container)
        
        self.setStyleSheet(MACOS_THEME)

        self.main_layout = QHBoxLayout(self.container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Sidebar
        self.sidebar = QFrame()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(220)
        self.setup_sidebar()
        self.main_layout.addWidget(self.sidebar)

        # Content
        self.content_area = QFrame()
        self.content_area.setObjectName("Content")
        self.content_layout = QVBoxLayout(self.content_area)
        self.content_layout.setContentsMargins(30, 20, 30, 30)
        
        self.setup_title_bar()
        
        self.pages = QStackedWidget()
        self.page_dashboard = QWidget()
        self.page_manual = QWidget()
        
        self.setup_dashboard_ui()
        self.setup_manual_ui()
        
        self.pages.addWidget(self.page_dashboard)
        self.pages.addWidget(self.page_manual)
        self.content_layout.addWidget(self.pages)
        self.main_layout.addWidget(self.content_area)

        # Logic
        self.monitor_thread = MonitorThread()
        self.monitor_thread.alert_signal.connect(self.show_alert)
        self.popup = DynamicIsland()
        self.manual_model = xgb.XGBClassifier()
        self.manual_model.load_model(os.path.join(MODELS_DIR, "url_classifier.json"))
        self.threat_count = 0
        self.old_pos = None

    def setup_title_bar(self):
        title_bar = QHBoxLayout()
        title_bar.setContentsMargins(0, 0, 0, 10)
        title_bar.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        btn_close = QPushButton()
        btn_close.setObjectName("CloseBtn")
        btn_close.setFixedSize(12, 12)
        btn_close.clicked.connect(self.close)

        btn_min = QPushButton()
        btn_min.setObjectName("MinBtn")
        btn_min.setFixedSize(12, 12)
        btn_min.clicked.connect(self.showMinimized)

        btn_max = QPushButton()
        btn_max.setObjectName("MaxBtn")
        btn_max.setFixedSize(12, 12)

        title_bar.addWidget(btn_close)
        title_bar.addSpacing(8)
        title_bar.addWidget(btn_min)
        title_bar.addSpacing(8)
        title_bar.addWidget(btn_max)
        title_bar.addWidget(QLabel("  ShieldAI Protection"))
        self.content_layout.addLayout(title_bar)

    def setup_sidebar(self):
        layout = QVBoxLayout(self.sidebar)
        layout.setContentsMargins(15, 30, 15, 30)
        layout.setSpacing(10)
        
        lbl = QLabel("ShieldAI")
        lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: white; margin-bottom: 20px;")
        layout.addWidget(lbl)

        self.btn_dash = QPushButton("  Dashboard")
        self.btn_dash.setObjectName("SidebarBtn")
        self.btn_dash.setCheckable(True)
        self.btn_dash.setChecked(True)
        self.btn_dash.clicked.connect(lambda: self.switch_page(0))
        
        self.btn_scan = QPushButton("  Deep Scan")
        self.btn_scan.setObjectName("SidebarBtn")
        self.btn_scan.setCheckable(True)
        self.btn_scan.clicked.connect(lambda: self.switch_page(1))

        layout.addWidget(self.btn_dash)
        layout.addWidget(self.btn_scan)
        layout.addStretch()

    def setup_dashboard_ui(self):
        layout = QVBoxLayout(self.page_dashboard)
        layout.setSpacing(20)

        stats_layout = QHBoxLayout()
        self.card_threats = self.create_card("Threats Blocked", "0", "#FF453A")
        self.card_status = self.create_card("Status", "Standby", "#FF9F0A")
        stats_layout.addWidget(self.card_threats)
        stats_layout.addWidget(self.card_status)
        layout.addLayout(stats_layout)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Protection")
        self.btn_start.setObjectName("ActionBtn")
        self.btn_start.clicked.connect(self.start_monitoring)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setObjectName("ActionBtn")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_monitoring)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        layout.addWidget(QLabel("Live Event Stream:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

    def setup_manual_ui(self):
        layout = QVBoxLayout(self.page_manual)
        layout.setSpacing(15)

        layout.addWidget(QLabel("Enter URL for Analysis:"))
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("https://suspicious-link.com...")
        layout.addWidget(self.url_input)

        self.btn_analyze = QPushButton("Analyze Now")
        self.btn_analyze.setObjectName("ActionBtn")
        self.btn_analyze.clicked.connect(self.analyze_manual_url)
        layout.addWidget(self.btn_analyze)

        # Removed Chart, kept Result Label
        self.result_label = QLabel("Waiting for input...")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #666; margin-top: 30px;")
        layout.addWidget(self.result_label)
        
        layout.addStretch()

    def create_card(self, title, value, color):
        frame = QFrame()
        frame.setObjectName("Card")
        l = QVBoxLayout(frame)
        t = QLabel(title)
        t.setStyleSheet("color: #888; font-size: 12px;")
        v = QLabel(value)
        v.setStyleSheet(f"color: {color}; font-size: 28px; font-weight: bold;")
        l.addWidget(t)
        l.addWidget(v)
        if title == "Threats Blocked": self.lbl_threat_val = v
        if title == "Status": self.lbl_status_val = v
        return frame

    def switch_page(self, index):
        self.pages.setCurrentIndex(index)
        self.btn_dash.setChecked(index == 0)
        self.btn_scan.setChecked(index == 1)

    def start_monitoring(self):
        self.monitor_thread.start()
        self.lbl_status_val.setText("Active")
        self.lbl_status_val.setStyleSheet("color: #30D158; font-size: 28px; font-weight: bold;")
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_monitoring(self):
        self.monitor_thread.stop()
        self.lbl_status_val.setText("Paused")
        self.lbl_status_val.setStyleSheet("color: #FF9F0A; font-size: 28px; font-weight: bold;")
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def show_alert(self, alert_type, message):
        self.threat_count += 1
        self.lbl_threat_val.setText(str(self.threat_count))
        timestamp = time.strftime("%H:%M:%S")
        color = "#FF453A" if "PHISHING" in alert_type else "#FFD60A"
        self.log_box.append(f'<span style="color:#555;">[{timestamp}]</span> <span style="color:{color}">{alert_type}:</span> {message}')
        self.popup.show_message(alert_type, message, color)

    def analyze_manual_url(self):
        url = self.url_input.text()
        if not url: return
        feats = get_url_features(url)
        if not feats: return
        df = pd.DataFrame([feats], columns=feature_names)
        prob = self.manual_model.predict_proba(df)[0][1]
        is_phishing = prob > 0.5
        
        text = "Danger Detected" if is_phishing else "Safe"
        color = "#FF453A" if is_phishing else "#30D158"
        self.result_label.setText(f"{text} ({prob*100:.1f}%)")
        self.result_label.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold;")

    # Window Dragging
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.old_pos = event.globalPosition().toPoint()
    def mouseMoveEvent(self, event):
        if self.old_pos:
            delta = event.globalPosition().toPoint() - self.old_pos
            self.move(self.pos() + delta)
            self.old_pos = event.globalPosition().toPoint()
    def mouseReleaseEvent(self, event):
        self.old_pos = None