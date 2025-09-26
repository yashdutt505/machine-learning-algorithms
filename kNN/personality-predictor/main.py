import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"

def save_artifacts(model, scaler, le):
    """Save trained model, scaler, and label encoder"""
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(le, ENCODER_PATH)

def load_artifacts():
    """Load trained model, scaler, and label encoder"""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, scaler, le


# Load and prepare data
def load_and_prepare_data(file_path):
    """Load and clean the personality data"""
    # Read CSV with proper handling
    df = pd.read_csv(file_path)
    
    # Convert numeric columns to proper types
    numeric_cols = ['Age', 'Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Rename 'Personality' to 'Personality Type' if needed
    if 'Personality' in df.columns:
        df.rename(columns={'Personality': 'Personality Type'}, inplace=True)
    
    # Handle Gender encoding
    df['Gender'] = (df['Gender'] == 'Male').astype(int)
    
    # Handle Education (already binary 0/1)
    df['Education'] = df['Education'].astype(int)
    
    # One-hot encode Interest
    df = pd.get_dummies(df, columns=['Interest'], prefix='Interest')
    
    # Store personality types before encoding for analysis
    personality_distribution = df['Personality Type'].value_counts()
    
    # Encode personality types
    le = LabelEncoder()
    df['Personality Type'] = le.fit_transform(df['Personality Type'])
    
    return df, le

def train_and_evaluate_model(X_train, y_train, X_test, y_test, le):
    """Train model"""
    
    best_k = 14
    model = KNeighborsClassifier(n_neighbors=best_k)
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    
    print(f"\nknn Accuracy: {accuracy:.4f}")
    print("-" * 30)
    print("Classification Report:")
    
    # Get personality type names for better interpretation
    target_names = [f"{i}:{name}" for i, name in enumerate(le.classes_)]
    print(classification_report(y_test, y_pred, target_names=target_names))

    return model

def main(file_path='data.csv'):
    """Main execution function"""
    # 1. Load and prepare data
    df, le = load_and_prepare_data(file_path)
    
    # 2. Prepare features and target
    X = df.drop('Personality Type', axis=1).values
    y = df['Personality Type'].values
    
    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 4. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Handle class imbalance if needed
    unique, counts = np.unique(y_train, return_counts=True)
    
    # Check if we need to balance classes
    if max(counts) / min(counts) > 2:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    
    # 6. Train and evaluate model
    model = train_and_evaluate_model(
        X_train_balanced, y_train_balanced, 
        X_test_scaled, y_test, le
    )
    
    # 7. Save the artifacts
    save_artifacts(model, scaler, le)
    return model, scaler, le


# Run the analysis
if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        print("🔄 Loading saved model...")
        model, scaler, le = load_artifacts()
    else:
        print("🚀 Training new model...")
        model, scaler, le = main('data.csv')