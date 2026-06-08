import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import train_model  # We reuse the loading logic to see what the model sees

# Configuration
DATA_PATH = 'indian_liver_patient.csv'
OUTPUT_DIR = 'eda_plots'

def run_eda():
    """Generates charts and statistics for the Project Report."""

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    # ─────────────────────────────────────────────
    # 1. MISSING VALUES HEATMAP (Raw Data)
    # ─────────────────────────────────────────────
    print("--- 1. Raw Data Analysis ---")
    df_raw = pd.read_csv(DATA_PATH)

    plt.figure(figsize=(10, 6))
    sns.heatmap(df_raw.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Values Heatmap (Raw Data)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/1_missing_values_raw.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/1_missing_values_raw.png")

    # ─────────────────────────────────────────────
    # 2. LOAD PROCESSED DATA
    # ─────────────────────────────────────────────
    print("\n--- 2. Processed Data Analysis ---")
    df_clean = train_model.load_and_preprocess_data(DATA_PATH)

    # ─────────────────────────────────────────────
    # 3. CLASS BALANCE
    # FIX: Added hue= and legend=False to remove FutureWarning
    # FIX: Added count labels on top of bars
    # ─────────────────────────────────────────────
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(
        x='Dataset', data=df_clean,
        hue='Dataset', palette='pastel',   # hue fixes FutureWarning
        order=[0, 1], legend=False
    )
    # Fix tick label warning by setting ticks first
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Healthy (0)', 'Disease (1)'])

    # Add count labels on top of each bar
    for p in ax.patches:
        ax.annotate(
            f'{int(p.get_height())}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )
    plt.title("Class Distribution (0=Healthy, 1=Disease)", fontsize=13)
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/2_class_balance.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/2_class_balance.png")

    # ─────────────────────────────────────────────
    # 4. CORRELATION MATRIX
    # FIX: Added numeric_only=True to remove FutureWarning
    # ─────────────────────────────────────────────
    plt.figure(figsize=(12, 10))
    correlation = df_clean.corr(numeric_only=True)  # FutureWarning fix
    sns.heatmap(
        correlation, annot=True, cmap='coolwarm',
        fmt=".2f", linewidths=0.5,
        annot_kws={"size": 9}
    )
    plt.title("Feature Correlation Matrix (Pearson)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/3_correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/3_correlation_matrix.png")

    # ─────────────────────────────────────────────
    # 5. PAIRPLOT
    # FIX: Convert Dataset to string so hue legend shows
    #      "Healthy" / "Disease" instead of 0 / 1
    # ─────────────────────────────────────────────
    key_features = ['Total_Bilirubin', 'Alkaline_Phosphotase', 'Age', 'Albumin', 'Dataset']
    df_pair = df_clean[key_features].copy()
    df_pair['Dataset'] = df_pair['Dataset'].map({0: 'Healthy', 1: 'Disease'})

    sns.pairplot(
        df_pair, hue='Dataset',
        palette={'Healthy': '#66b3ff', 'Disease': '#ff9999'},
        plot_kws={'alpha': 0.5, 's': 20},
        diag_kind='kde'
    )
    plt.suptitle("Pairplot of Key Clinical Features by Disease Status",
                 y=1.02, fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/4_pairplot_key_features.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/4_pairplot_key_features.png")

    # ─────────────────────────────────────────────
    # 6. AGE DISTRIBUTION
    # FIX: Map Dataset 0/1 to labels for cleaner legend
    # ─────────────────────────────────────────────
    df_age = df_clean.copy()
    df_age['Status'] = df_age['Dataset'].map({0: 'Healthy', 1: 'Disease'})

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df_age, x='Age', hue='Status',
        kde=True, element="step",
        palette={'Healthy': '#66b3ff', 'Disease': '#ff9999'},
        alpha=0.6
    )
    plt.title("Age Distribution by Disease Status", fontsize=14)
    plt.xlabel("Age (Years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/5_age_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/5_age_distribution.png")

    # ─────────────────────────────────────────────
    # 7. BONUS: BOXPLOTS — Key Biomarkers by Disease Status
    #    Shows outlier distribution for the 4 most important features
    # ─────────────────────────────────────────────
    top_features = ['Total_Bilirubin', 'Alkaline_Phosphotase',
                    'Aspartate_Aminotransferase', 'Albumin']
    df_box = df_clean.copy()
    df_box['Status'] = df_box['Dataset'].map({0: 'Healthy', 1: 'Disease'})

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    for ax, feat in zip(axes, top_features):
        sns.boxplot(
            data=df_box, x='Status', y=feat,
            hue='Status',
            palette={'Healthy': '#66b3ff', 'Disease': '#ff9999'},
            ax=ax, legend=False, showfliers=True,
            flierprops=dict(marker='o', markersize=3, alpha=0.4)
        )
        ax.set_title(feat.replace('_', ' '), fontsize=11)
        ax.set_xlabel('')

    plt.suptitle("Key Biomarker Distributions by Disease Status", fontsize=13)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/6_biomarker_boxplots.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR}/6_biomarker_boxplots.png")

    # ─────────────────────────────────────────────
    # PRINT SUMMARY STATISTICS
    # ─────────────────────────────────────────────
    print("\n--- Dataset Summary Statistics ---")
    print(f"Total records    : {len(df_clean)}")
    print(f"Disease patients : {(df_clean['Dataset']==1).sum()} "
          f"({(df_clean['Dataset']==1).mean()*100:.1f}%)")
    print(f"Healthy patients : {(df_clean['Dataset']==0).sum()} "
          f"({(df_clean['Dataset']==0).mean()*100:.1f}%)")
    print(f"Missing values   : {df_raw.isnull().sum().sum()} "
          f"(in Albumin_and_Globulin_Ratio only)")

    print(f"\n✅ EDA Complete. All plots saved in '{OUTPUT_DIR}' folder.")
    print("Plots generated:")
    print("  1_missing_values_raw.png   — Data quality validation")
    print("  2_class_balance.png        — Class imbalance visualization")
    print("  3_correlation_matrix.png   — Pearson feature correlations")
    print("  4_pairplot_key_features.png — Feature pair relationships")
    print("  5_age_distribution.png     — Age stratification by diagnosis")
    print("  6_biomarker_boxplots.png   — Outlier distribution analysis (BONUS)")

if __name__ == "__main__":
    run_eda()
