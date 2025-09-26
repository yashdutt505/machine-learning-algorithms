# 🧠 Personality Predictor (KNN + Random Forest)

This project predicts personality type based on demographic and psychological traits.  
It uses **K-Nearest Neighbors (KNN)**.

---

## 🚀 Features
- Preprocessing with label encoding & one-hot encoding
- Handles imbalanced data using **SMOTE**
- Saves trained model, scaler, and label encoder
- Easy prediction script (`predict.py`)

---

## 📂 Project Structure
personality-predictor/
├── main.py # Train and save model
├── predict.py # Load model and make predictions
├── test.py # Quick test script
├── model.pkl # Trained model
├── scaler.pkl # Scaler for features
├── label_encoder.pkl # Encoder for target labels
├── requirements.txt # Dependencies
└── README.md # Documentation

## ⚙️ Installation
```bash
git clone https://github.com/your-username/personality-predictor.git
cd personality-predictor
pip install -r requirements.txt
🏃 Usage
1. Train Model
bash
Copy code
python main.py
2. Predict Personality
bash
Copy code
python predict.py
Example:
Predicted Personality: INTJ
📈 Future Improvements
Deploy as a web app with Flask/Django

Collect larger, real-world dataset

Experiment with advanced models (XGBoost, Neural Nets)

👨‍💻 Author

Developed by Yash Dutt
