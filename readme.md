🛡️ Project: ShieldAI - Real-Time Phishing & Scam Detection System

1. Project Description
ShieldAI is an intelligent, automated cybersecurity tool that protects users from phishing attacks in real-time. Unlike traditional antiviruses that scan files, ShieldAI "watches" the screen using Computer Vision (OCR). It detects malicious URLs and scam messages (SMS/Chat) instantly and alerts the user via a non-intrusive "Dynamic Island" style popup. It uses Explainable AI (SHAP) to show users why a link was flagged.

2. Technology Stack & Libraries
![alt text](image.png)

3. Models & Algorithms
We use a Dual-Model Architecture (Two separate brains working together).

Model A: URL Phishing Detector
Algorithm: XGBoost Classifier (Extreme Gradient Boosting).
Why XGBoost? It is the industry standard for tabular data classification, offering higher speed and accuracy than Neural Networks for this specific task.
Input Features: 12+ extracted mathematical features (e.g., URL length, count of dots, presence of IP address, suspicious keywords like 'login').
Accuracy: ~78-85% (depending on dataset size).

Model B: Scam Message Detector
Algorithm: Random Forest Classifier.
Feature Engineering: TF-IDF Vectorizer (Term Frequency-Inverse Document Frequency).
Logic: Converts words into statistical weights. Words like "Urgent", "Winner", and "Bank" get high "suspicion scores."
Accuracy: ~98% (Highly effective on short text).

5. Workflow & Logic Flowchart
The Logic Loop (Runs every 2 seconds):

Capture: mss takes a screenshot of the active window.

Diff Check (Optimization): The system compares the new screenshot with the previous one using MSE (Mean Squared Error).
If Difference < 1%: SLEEP (Save CPU).
If Difference > 1%: PROCEED.

OCR: pytesseract extracts all visible text.

Route Data:
Is it a Link? -> Extract 12 features -> Send to XGBoost.
Is it a Sentence? -> Vectorize (TF-IDF) -> Send to Random Forest.

Action:
If Safe: Do nothing.
If Threat: Trigger DynamicIsland popup and log to Dashboard.

6. File Usage Description
Here is the breakdown of every file in your project folder:

Root Folder
main.py: The Entry Point. It sets up the system paths and launches the GUI. Always run this file to start the app.
requirements.txt: Contains the list of all libraries needed (pip install -r requirements.txt).
src/ Folder (Source Code)
gui_main.py: The Master Interface. Contains the code for the Dashboard, Sidebar, Threading logic, and Dark/Glass themes. It connects the visual UI to the backend logic.
notifications.py: The Visual Alert. Contains the class for the "Dynamic Island" popup—handling the animation, blur effect, and top-layer positioning.
feature_extractor.py: The Translator. Contains the math logic that converts a raw URL string (e.g., google.com) into numbers (e.g., [10, 0, 0, 0...]) for the AI.
train_models.py: The Teacher. Run this once to read the CSV datasets and generate the "Brain" files (.json, .pkl).
evaluate_models.py: The Report Card. Calculates accuracy, precision, and generates Confusion Matrix graphs to prove the model works.
merge_data.py: The Helper. Used to combine multiple CSV files into one master dataset if you download more data later.
models/ Folder
url_classifier.json: The trained XGBoost brain.
nlp_model.pkl: The trained Random Forest brain.
vectorizer.pkl: The dictionary used to understand English words.
accuracy/ Folder
Stores your generated performance reports and graphs (.png files).