# main.py
import sys
import os

# 1. Get the path to the 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')

# 2. Add 'src' to Python's search path BEFORE importing anything else
sys.path.append(src_path)

# 3. Now we can import from the src folder
from gui_main import PhishingApp
from PyQt6.QtWidgets import QApplication

# 4. Run the App
if __name__ == "__main__":
    print("--- Starting Phishing Detection System ---")
    app = QApplication(sys.argv)
    window = PhishingApp()
    window.show()
    sys.exit(app.exec())