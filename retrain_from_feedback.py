import pandas as pd
import sqlite3
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import train_model  # Import preprocessing logic to ensure consistency

DB_PATH = 'feedback.db'
ORIGINAL_DATA_PATH = 'indian_liver_patient.csv'
MODEL_PATH = 'liver_model.pkl'

def get_feedback_data():
    """Fetches verified human labels from the database."""
    conn = sqlite3.connect(DB_PATH)
    # We only fetch data where we have a human label
    query = "SELECT * FROM liver_feedback"
    try:
        feedback_df = pd.read_sql(query, conn)
        conn.close()
        
        if feedback_df.empty:
            return None
            
        # Parse the stored input features
        # The DB stores inputs as columns, ensuring they match the training data
        # We need to drop metadata columns (id, timestamp, ai_prediction, etc.)
        drop_cols = ['id', 'timestamp', 'ai_prediction', 'human_label', 'agreement']
        
        X_feedback = feedback_df.drop(columns=[c for c in drop_cols if c in feedback_df.columns])
        y_feedback = feedback_df['human_label']
        
        return X_feedback, y_feedback
        
    except Exception as e:
        print(f"Error reading DB: {e}")
        conn.close()
        return None

def retrain_system():
    """Combines original data with feedback data and retrains the model."""
    
    # 1. Load Original Data
    print("Loading original dataset...")
    df_orig = train_model.load_and_preprocess_data(ORIGINAL_DATA_PATH)
    X_orig = df_orig.drop('Dataset', axis=1)
    y_orig = df_orig['Dataset']
    
    # 2. Load Feedback Data
    print("Loading feedback data...")
    feedback_data = get_feedback_data()
    
    if feedback_data is None:
        return "No feedback data available to retrain."
    
    X_new, y_new = feedback_data
    
    # Ensure columns match
    # (In a real scenario, we'd add rigorous schema checking here)
    X_new = X_new[X_orig.columns] 
    
    # 3. Combine Datasets
    X_combined = pd.concat([X_orig, X_new], axis=0)
    y_combined = pd.concat([y_orig, y_new], axis=0)
    
    print(f"Retraining on {len(X_combined)} total samples ({len(X_new)} new from feedback).")
    
    # 4. Retrain
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_combined, y_combined)
    
    # 5. Save
    joblib.dump(model, MODEL_PATH)
    
    return f"Success! Model retrained on {len(X_combined)} records."

if __name__ == "__main__":
    print(retrain_system())