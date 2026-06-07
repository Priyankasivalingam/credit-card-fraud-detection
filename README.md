# credit-card-fraud-detection
💳 Credit Card Fraud Detection System
Show Image
Show Image
Show Image
Show Image

Real-time and bulk credit card fraud detection powered by class-imbalance-aware Machine Learning.


📌 Overview
A production-ready fraud detection system that identifies fraudulent credit card transactions with high precision. The model handles severe class imbalance (a real-world challenge in fraud datasets) using advanced resampling techniques and delivers sub-second inference through an intuitive Streamlit dashboard with bulk CSV analysis and downloadable reports.

✨ Features

⚖️ Class-Imbalance Handling — SMOTE / undersampling techniques for balanced training
🔄 Full Preprocessing Pipeline — Scaling, encoding, and feature selection automated
📁 Bulk CSV Analysis — Upload thousands of transactions and analyze in one click
📥 Downloadable Reports — Export results as CSV with fraud probability scores
⚡ Sub-second Inference — Optimized model serving via Joblib
📊 Interactive Dashboard — Visualize fraud distribution, confidence scores, and transaction stats


🏗️ Architecture
Raw Transaction Data (CSV / Single Input)
             │
             ▼
  Preprocessing Pipeline
  (Scaling → Encoding → Feature Selection)
             │
             ▼
  Class-Imbalance Handling (SMOTE)
             │
             ▼
  Random Forest Classifier
             │
        ┌────┴────┐
        ▼         ▼
   Fraud Score  Prediction Label
        │         │
        └────┬────┘
             ▼
    Streamlit Dashboard
    (Results + Charts + CSV Export)

🧰 Tech Stack
LayerTechnologyML ModelRandom Forest (Scikit-learn)Imbalance HandlingSMOTE (imbalanced-learn)Model PersistenceJoblibUIStreamlitData ProcessingPandas, NumPyLanguagePython 3.9+

🚀 Getting Started
Prerequisites
bashPython 3.9+
pip
Installation
bash# Clone the repository
git clone https://github.com/Priyankasivalingam/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
Requirements
scikit-learn
streamlit
pandas
numpy
joblib
imbalanced-learn
matplotlib
seaborn


🖥️ Usage
Single Transaction

Open the app → navigate to Single Transaction tab
Enter transaction details in the form
Click Predict → view fraud probability and verdict

Bulk CSV Upload

Navigate to Bulk Analysis tab
Upload your .csv file (must match expected columns)
View results table with fraud scores
Click Download Report to export


📊 Model Performance
MetricScoreAccuracy~99%Precision (Fraud)~95%Recall (Fraud)~93%F1-Score (Fraud)~94%ROC-AUC~97%

Evaluated on the standard Kaggle credit card fraud dataset.


🔮 Future Improvements

 Real-time transaction stream integration (Kafka)
 Explainability layer (SHAP values)
 REST API endpoint for integration with banking systems
 Email/SMS alert system for high-confidence fraud


👩‍💻 Author
Priyanka S — LinkedIn · GitHub

📄 License
This project is licensed under the MIT License.
