import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib
import os

# --- CONFIGURATION ---
DATA_PATH = 'indian_liver_patient.csv'
MODEL_PATH = 'liver_model.pkl'
FEATURES_PATH = 'feature_names.pkl'
METRICS_PATH = 'model_metrics.pkl'
BG_DATA_PATH = 'x_train_bg.pkl'  # Required for LIME Explainer

def load_and_preprocess_data(filepath):
    """Loads and cleans the dataset."""
    print(f"Loading dataset from {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}. Please ensure the CSV file is in the folder.")
    
    df = pd.read_csv(filepath)
    
    # 1. Handle Missing Values (Albumin_and_Globulin_Ratio)
    if df.isnull().sum().any():
        print("Imputing missing values...")
        imputer = SimpleImputer(strategy='mean')
        df['Albumin_and_Globulin_Ratio'] = imputer.fit_transform(df[['Albumin_and_Globulin_Ratio']])
    
    # 2. Encode Gender (Male -> 1, Female -> 0)
    print("Encoding categorical variables...")
    df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
    
    # 3. Encode Target (1=Disease, 2=Healthy -> 1=Disease, 0=Healthy)
    # This standardizes the target for standard ML metrics
    df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
    
    return df

def train_model():
    """Trains multiple models, benchmarks them, and saves the best one."""
    
    # --- 1. PREPARATION ---
    df = load_and_preprocess_data(DATA_PATH)
    X = df.drop('Dataset', axis=1)
    y = df['Dataset']
    
    # Save feature names for the App
    joblib.dump(X.columns.tolist(), FEATURES_PATH)
    
    # Split Data (Stratified to maintain disease/healthy ratio)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # CRITICAL: Save X_train for LIME (Background dataset)
    # LIME needs this to understand what "normal" data looks like
    joblib.dump(X_train, BG_DATA_PATH)
    
    # --- 2. MULTI-MODEL COMPARISON ---
    print("\nTraining Multiple Models for Benchmarking...")
    
    # We use make_pipeline(StandardScaler(), Model) for Logistic Regression and SVM.
    # This scales the data (normalizes it) which prevents the ConvergenceWarning 
    # and improves accuracy for these specific algorithms.
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "SVM (Support Vector)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=42))
    }
    
    comparison_metrics = []
    main_model = None
    main_cm = None
    main_acc = 0
    
    # Print header for live updates
    print(f"\n{'Algorithm':<25} | {'Accuracy':<10} | {'F1-Score':<10}")
    print("-" * 50)

    for name, clf in models.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Calculate Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Print results immediately
        print(f"{name:<25} | {acc*100:.1f}%     | {f1*100:.1f}%")

        comparison_metrics.append({
            "Algorithm": name,
            "Accuracy": f"{acc*100:.1f}%",
            "Precision": f"{prec*100:.1f}%",
            "Recall": f"{rec*100:.1f}%",
            "F1-Score": f"{f1*100:.1f}%"
        })
        
        # We select Random Forest as the 'Deployed' model because:
        # 1. It has high accuracy
        # 2. It works natively with SHAP TreeExplainer (fast XAI)
        if name == "Random Forest":
            main_model = clf
            main_cm = confusion_matrix(y_test, y_pred)
            main_acc = acc

    print("-" * 50)

    # --- 3. SAVE ARTIFACTS ---
    print(f"\nSaving Main Model (Random Forest)...")
    joblib.dump(main_model, MODEL_PATH)
    
    # Save the comparison table and other metrics for the Dashboard
    metrics_package = {
        'main_accuracy': main_acc,
        'confusion_matrix': main_cm,
        'comparison_table': comparison_metrics
    }
    joblib.dump(metrics_package, METRICS_PATH)
    
    print(f"✅ Training Complete.")
    print(f"Files Generated: {MODEL_PATH}, {METRICS_PATH}, {BG_DATA_PATH}")

if __name__ == "__main__":
    train_model()