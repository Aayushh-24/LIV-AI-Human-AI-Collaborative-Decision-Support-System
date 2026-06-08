import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier  # <-- needed for fresh train
import os

print("Generating high-quality plots for IEEE Research Paper...")

# ─────────────────────────────────────────────────────────────────
# SHARED PREPROCESSING (used by ROC + all plots consistently)
# ─────────────────────────────────────────────────────────────────
imputer = SimpleImputer(strategy='mean')
df_raw = pd.read_csv('indian_liver_patient.csv')
df_raw['Albumin_and_Globulin_Ratio'] = imputer.fit_transform(
    df_raw[['Albumin_and_Globulin_Ratio']])
df_raw['Gender'] = df_raw['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df_raw['Dataset'] = df_raw['Dataset'].map({1: 1, 2: 0})

X = df_raw.drop('Dataset', axis=1)
y = df_raw['Dataset']

# CRITICAL: Same random_state=42 and stratify=y as train_model.py
# This ensures X_test here is IDENTICAL to the one used during original training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ─────────────────────────────────────────────────────────────────
# BUG FIX: Train a FRESH model for plot generation
# DO NOT load liver_model.pkl — that file may have been retrained
# by the Active Learning module (retrain_from_feedback.py) and
# will produce inflated/incorrect evaluation metrics.
# We retrain here with the SAME hyperparameters as the original
# train_model.py to get honest, reproducible evaluation results.
# ─────────────────────────────────────────────────────────────────
print("\nTraining fresh model for honest evaluation plots...")
plot_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=10
)
plot_model.fit(X_train, y_train)
print("✅ Fresh model trained successfully.")

# ─────────────────────────────────────────────────────────────────
# 1. DATASET CLASS DISTRIBUTION
# ─────────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
ax = sns.countplot(x='Dataset', data=df_raw, palette='pastel',
                   order=[0, 1])
ax.set_xticklabels(['Healthy (0)', 'Disease (1)'])
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
plt.title("Class Distribution (0=Healthy, 1=Disease)", fontsize=13)
plt.xlabel("Patient Status")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig('dataset_distribution.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ Saved: dataset_distribution.png")

# ─────────────────────────────────────────────────────────────────
# 2. CONFUSION MATRIX (from model_metrics.pkl — original training)
# ─────────────────────────────────────────────────────────────────
metrics = joblib.load('model_metrics.pkl')
cm = metrics['confusion_matrix']

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Disease'],
            yticklabels=['Healthy', 'Disease'],
            linewidths=1, linecolor='gray')
plt.title("Random Forest Confusion Matrix", fontsize=13)
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.tight_layout()
plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ Saved: confusion_matrix.png")

# ─────────────────────────────────────────────────────────────────
# 3. MODEL COMPARISON BAR CHART (Accuracy + Recall side by side)
# ─────────────────────────────────────────────────────────────────
comp_df = pd.DataFrame(metrics['comparison_table'])
comp_df['Accuracy_Num'] = comp_df['Accuracy'].str.rstrip('%').astype(float)
comp_df['Recall_Num'] = comp_df['Recall'].str.rstrip('%').astype(float)

x = range(len(comp_df))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 5))
bars1 = ax.bar([i - width/2 for i in x], comp_df['Accuracy_Num'],
               width, label='Accuracy', color='steelblue', alpha=0.85)
bars2 = ax.bar([i + width/2 for i in x], comp_df['Recall_Num'],
               width, label='Recall', color='darkorange', alpha=0.85)

# Add value labels on bars
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)

ax.set_ylim(0, 115)
ax.set_xticks(list(x))
ax.set_xticklabels(comp_df['Algorithm'], rotation=12, ha='right')
ax.set_ylabel("Score (%)")
ax.set_title("Model Accuracy & Recall Comparison", fontsize=13)
ax.legend()
ax.axhline(y=73.5, color='steelblue', linestyle='--', alpha=0.4, linewidth=1)
plt.tight_layout()
plt.savefig('model_comparison.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ Saved: model_comparison.png")

# ─────────────────────────────────────────────────────────────────
# 4. ROC CURVE — CORRECTED
#
# ORIGINAL BUG in your code (line 47):
#   model = joblib.load('model_metrics.pkl')   <-- WRONG FILE LOADED
#   This loaded a dict, not a model. Python didn't crash because
#   the retrained liver_model.pkl happened to give AUC=1.00 due to
#   overfitting on the tiny 3-record feedback-augmented dataset.
#
# FIX: Use the freshly trained plot_model evaluated on X_test
# This gives an honest AUC on data the model has never seen.
# ─────────────────────────────────────────────────────────────────
y_probs = plot_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

print(f"\n📊 Honest AUC on held-out test set: {roc_auc:.4f}")
print(f"   (This is the correct value to report in your black book)")

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2.5,
         label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--',
         label='Random Classifier (AUC = 0.50)')
plt.fill_between(fpr, tpr, alpha=0.08, color='darkorange')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.title('ROC Curve — Random Forest Classifier\n(Evaluated on Held-Out Test Set)',
          fontsize=12)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ Saved: roc_curve.png")

# ─────────────────────────────────────────────────────────────────
# 5. BONUS: FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────────────────────────
feature_names = joblib.load('feature_names.pkl')
importances = pd.Series(
    plot_model.feature_importances_, index=feature_names).sort_values()

plt.figure(figsize=(8, 5))
colors = ['#d73027' if i > importances.median() else '#4575b4'
          for i in importances]
importances.plot(kind='barh', color=colors)
plt.xlabel('Feature Importance (Gini)', fontsize=11)
plt.title('Random Forest — Global Feature Importance', fontsize=13)
plt.axvline(x=importances.median(), color='gray',
            linestyle='--', alpha=0.6, label='Median importance')
plt.legend()
plt.tight_layout()
plt.savefig('feature_importance.png', bbox_inches='tight', dpi=300)
plt.close()
print("✅ Saved: feature_importance.png")

# ─────────────────────────────────────────────────────────────────
# SUMMARY REPORT
# ─────────────────────────────────────────────────────────────────
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
y_pred = plot_model.predict(X_test)
print("\n" + "="*50)
print("VERIFIED METRICS FOR BLACK BOOK (fresh model on test set):")
print("="*50)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred)*100:.1f}%")
print(f"  Precision : {precision_score(y_test, y_pred)*100:.1f}%")
print(f"  Recall    : {recall_score(y_test, y_pred)*100:.1f}%")
print(f"  F1-Score  : {f1_score(y_test, y_pred)*100:.1f}%")
print(f"  AUC-ROC   : {roc_auc:.4f}")
print("="*50)
print("\n🎉 All 5 images generated successfully!")
print("   Use roc_curve.png — the AUC=1.00 graph was a bug.")