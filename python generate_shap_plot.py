import pandas as pd
import shap
import matplotlib.pyplot as plt
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

print("Loading data and model for SHAP Feature Importance...")

# 1. Load and clean the data
df_raw = pd.read_csv('indian_liver_patient.csv')
imputer = SimpleImputer(strategy='mean')
df_raw['Albumin_and_Globulin_Ratio'] = imputer.fit_transform(df_raw[['Albumin_and_Globulin_Ratio']])
df_raw['Gender'] = df_raw['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df_raw['Dataset'] = df_raw['Dataset'].map({1: 1, 2: 0})

X = df_raw.drop('Dataset', axis=1)
y = df_raw['Dataset']

# 2. Get the exact X_test data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Load your trained model
model = joblib.load('liver_model.pkl')

print("Generating SHAP Feature Importance Bar Chart...")

# 4. Generate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# 5. THE FIX: Handle both old and new versions of the SHAP library safely
if isinstance(shap_values, list):
    # Older SHAP: It's a list, grab the array for class 1 (Disease)
    shap_values_disease = shap_values[1]
elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    # Newer SHAP: It's a 3D array, slice all rows, all columns, index 1 (Disease)
    shap_values_disease = shap_values[:, :, 1]
else:
    # Fallback just in case
    shap_values_disease = shap_values

# 6. Plot and save the chart
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values_disease, X_test, plot_type="bar", show=False, color="#d73027")

plt.title("SHAP Global Feature Importance (Mean Absolute Impact)", fontsize=13, pad=15)
plt.xlabel("mean(|SHAP value|) (average impact on model output magnitude)")
plt.tight_layout()

# Save the image
plt.savefig('shap_feature_importance_bar.png', bbox_inches='tight', dpi=300)
plt.close()

print("✅ Success! Saved: shap_feature_importance_bar.png")