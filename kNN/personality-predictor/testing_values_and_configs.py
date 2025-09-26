import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
def load_and_prepare_data(file_path):
    """Load and clean the personality data"""
    # Read CSV with proper handling
    df = pd.read_csv(file_path)
    
    # Print initial info
    print("Dataset shape:", df.shape)
    print("\nColumn names and types:")
    print(df.dtypes)
    print("\nFirst few rows:")
    print(df.head())
    
    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())
    
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
    print("\nPersonality Type Distribution:")
    print(personality_distribution)
    
    # Encode personality types
    le = LabelEncoder()
    df['Personality Type'] = le.fit_transform(df['Personality Type'])
    
    print("\nLabel mapping:")
    for i, label in enumerate(le.classes_):
        print(f"  {i}: {label}")
    
    return df, le

def analyze_data(df):
    """Perform exploratory data analysis"""
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Distribution of personality scores
    score_cols = ['Introversion Score', 'Sensing Score', 'Thinking Score', 'Judging Score']
    
    for idx, col in enumerate(score_cols):
        ax = axes[idx//2, idx%2]
        df.boxplot(column=col, by='Personality Type', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel('Personality Type (Encoded)')
    
    # Age distribution
    axes[0, 2].hist(df['Age'], bins=20, edgecolor='black')
    axes[0, 2].set_title('Age Distribution')
    axes[0, 2].set_xlabel('Age')
    axes[0, 2].set_ylabel('Frequency')
    
    # Gender distribution
    gender_counts = df['Gender'].value_counts()
    axes[1, 2].bar(['Female', 'Male'], gender_counts.values)
    axes[1, 2].set_title('Gender Distribution')
    axes[1, 2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.show()

def optimize_knn(X_train, y_train, X_test, y_test):
    """Find optimal K value for KNN"""
    print("\n" + "="*50)
    print("OPTIMIZING KNN CLASSIFIER")
    print("="*50)
    
    # Test different K values
    k_range = range(1, min(31, len(X_train)//2))
    k_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
        k_scores.append(scores.mean())
    
    # Plot K vs Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, k_scores, 'b-', marker='o')
    plt.xlabel('K Value')
    plt.ylabel('Cross-Validation Accuracy')
    plt.title('KNN: Optimal K Value Selection')
    plt.grid(True, alpha=0.3)
    
    best_k = k_range[np.argmax(k_scores)]
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Best K={best_k}')
    plt.legend()
    plt.show()
    
    print(f"Best K value: {best_k}")
    print(f"Best cross-validation accuracy: {max(k_scores):.4f}")
    
    return best_k

def train_and_evaluate_models(X_train, y_train, X_test, y_test, le):
    """Train multiple models and compare performance"""
    models = {}
    results = {}
    
    # 1. Optimized KNN
    best_k = optimize_knn(X_train, y_train, X_test, y_test)
    models['KNN'] = KNeighborsClassifier(n_neighbors=best_k)
    
    # 2. KNN with different distance metrics
    models['KNN_Manhattan'] = KNeighborsClassifier(n_neighbors=best_k, metric='manhattan')
    
    # 3. Weighted KNN
    models['KNN_Weighted'] = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    
    # 4. Random Forest (often better for this type of data)
    models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # 5. SVM
    models['SVM'] = SVC(kernel='rbf', random_state=42)
    
    print("\n" + "="*50)
    print("MODEL COMPARISON")
    print("="*50)
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'model': model
        }
        
        print(f"\n{name} Accuracy: {accuracy:.4f}")
        print("-" * 30)
        print("Classification Report:")
        
        # Get personality type names for better interpretation
        target_names = [f"{i}:{name}" for i, name in enumerate(le.classes_)]
        print(classification_report(y_test, y_pred, target_names=target_names))
    
    return results, models

def visualize_results(results, y_test, le):
    """Visualize model comparison and confusion matrices"""
    # Model comparison bar plot
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    bars = plt.bar(model_names, accuracies, color=['blue', 'green', 'red', 'purple', 'orange'])
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.ylim([0, 1])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Confusion matrices for top 2 models
    sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    for idx, (name, result) in enumerate(sorted_models[:2]):
        cm = confusion_matrix(y_test, result['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le.classes_, yticklabels=le.classes_,
                   ax=axes[idx])
        axes[idx].set_title(f'Confusion Matrix - {name}')
        axes[idx].set_ylabel('True Label')
        axes[idx].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

def main(file_path='data.csv'):
    """Main execution function"""
    # 1. Load and prepare data
    df, le = load_and_prepare_data(file_path)
    
    # 2. Analyze data
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    analyze_data(df)
    
    # 3. Prepare features and target
    X = df.drop('Personality Type', axis=1).values
    y = df['Personality Type'].values
    
    # 4. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # 5. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 6. Handle class imbalance if needed
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nClass distribution in training set:")
    for u, c in zip(unique, counts):
        print(f"  Class {u} ({le.classes_[u]}): {c} samples")
    
    # Check if we need to balance classes
    if max(counts) / min(counts) > 2:
        print("\nClass imbalance detected. Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE: {len(X_train_balanced)} training samples")
    else:
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train
    
    # 7. Train and evaluate models
    results, models = train_and_evaluate_models(
        X_train_balanced, y_train_balanced, 
        X_test_scaled, y_test, le
    )
    
    # 8. Visualize results
    visualize_results(results, y_test, le)
    
    # 9. Feature importance (for Random Forest)
    if 'Random Forest' in models:
        rf_model = models['Random Forest']
        feature_names = df.drop('Personality Type', axis=1).columns
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*50)
        print("TOP 10 MOST IMPORTANT FEATURES (Random Forest)")
        print("="*50)
        print(feature_importance.head(10))
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title('Top 10 Feature Importances')
        plt.tight_layout()
        plt.show()
    
    return models, results, scaler, le

# Run the analysis
if __name__ == "__main__":
    models, results, scaler, le = main('data.csv')