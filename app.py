import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import sqlite3
import base64
from datetime import datetime
from fpdf import FPDF
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import retrain_from_feedback  # Ensure this file is in the same folder

# --- CONFIGURATION ---
st.set_page_config(page_title="LIV-AI System", layout="wide", page_icon="🔬", initial_sidebar_state="expanded")

# --- ADVANCED CUSTOM CSS ---
def inject_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Poppins', sans-serif;
        }

        /* Gradient Header */
        .gradient-header {
            background: linear-gradient(135deg, #1E293B 0%, #0EA5E9 100%);
            padding: 30px;
            border-radius: 15px;
            color: white;
            text-align: center;
            box-shadow: 0 10px 30px rgba(14, 165, 233, 0.3);
            margin-bottom: 30px;
        }
        .gradient-header h1 { margin: 0; font-size: 3.5rem; font-weight: 800; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); }
        .gradient-header p { margin: 5px 0 0 0; font-size: 1.2rem; opacity: 0.9; font-weight: 300; }

        /* Bright Diagnostic Cards */
        .risk-card {
            background: linear-gradient(135deg, #FF416C 0%, #FF4B2B 100%);
            padding: 30px; border-radius: 20px; color: white; text-align: center;
            box-shadow: 0 15px 35px rgba(255, 75, 43, 0.4);
            transition: transform 0.3s ease;
        }
        .risk-card:hover { transform: translateY(-5px); }
        
        .safe-card {
            background: linear-gradient(135deg, #11998E 0%, #38EF7D 100%);
            padding: 30px; border-radius: 20px; color: white; text-align: center;
            box-shadow: 0 15px 35px rgba(56, 239, 125, 0.4);
            transition: transform 0.3s ease;
        }
        .safe-card:hover { transform: translateY(-5px); }
        
        /* Uncertainty Card */
        .info-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(200, 200, 200, 0.3);
            padding: 25px; border-radius: 20px; text-align: center;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            height: 100%;
        }
        
        /* Button Styling */
        div.stButton > button {
            background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%);
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 10px 20px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(118, 75, 162, 0.3);
        }
        div.stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(118, 75, 162, 0.5);
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# --- FILE PATHS ---
MODEL_PATH = 'liver_model.pkl'
FEATURES_PATH = 'feature_names.pkl'
METRICS_PATH = 'model_metrics.pkl'
BG_DATA_PATH = 'x_train_bg.pkl'
DB_PATH = 'feedback.db'

# --- SESSION STATE INITIALIZATION ---
if 'prediction_made' not in st.session_state:
    st.session_state['prediction_made'] = False

keys = ['last_input', 'last_input_df', 'last_pred', 'last_probs', 'last_std']
for key in keys:
    if key not in st.session_state:
        st.session_state[key] = None

# --- DATABASE MANAGEMENT ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS liver_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            Age INTEGER,
            Gender INTEGER,
            Total_Bilirubin REAL,
            Direct_Bilirubin REAL,
            Alkaline_Phosphotase INTEGER,
            Alamine_Aminotransferase INTEGER,
            Aspartate_Aminotransferase INTEGER,
            Total_Protiens REAL,
            Albumin REAL,
            Albumin_and_Globulin_Ratio REAL,
            ai_prediction INTEGER,
            human_label INTEGER,
            agreement TEXT,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def save_feedback(input_data, ai_pred, human_label):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    agreement = "Yes" if ai_pred == human_label else "No"
    
    data = list(input_data.values())
    data.extend([int(ai_pred), int(human_label), agreement, datetime.now()])
    
    placeholders = ', '.join(['?'] * len(data))
    columns = ', '.join(input_data.keys()) + ", ai_prediction, human_label, agreement, timestamp"
    
    query = f"INSERT INTO liver_feedback ({columns}) VALUES ({placeholders})"
    c.execute(query, data)
    conn.commit()
    conn.close()

# --- PDF GENERATOR ---
def create_pdf(input_data, prediction, confidence, ci_text, reliability):
    clean_reliability = reliability.replace("🟢", "").replace("🟡", "").replace("🔴", "").replace("✨", "").strip()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(200, 10, txt="LIV-AI Clinical Decision Support Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="[ PATIENT VITALS ]", ln=True)
    pdf.set_font("Arial", size=12)
    for key, value in input_data.items():
        val = "Male" if key == "Gender" and value == 1 else "Female" if key == "Gender" else value
        pdf.cell(200, 8, txt=f"  > {key}: {val}", ln=True)
    
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    if prediction == 1:
        pdf.set_text_color(220, 38, 38) # Red
        status = "HIGH RISK (LIVER DISEASE DETECTED)"
    else:
        pdf.set_text_color(22, 163, 74) # Green
        status = "LOW RISK (HEALTHY PROFILE)"
        
    pdf.cell(200, 10, txt=f"AI DIAGNOSIS: {status}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Base Confidence Score: {confidence}", ln=True)
    pdf.cell(200, 10, txt=f"95% Confidence Interval: {ci_text}", ln=True)
    pdf.cell(200, 10, txt=f"System Reliability: {clean_reliability}", ln=True)
    
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(200, 10, txt="Disclaimer: This report is generated by an AI assistant and must be verified by a certified hepatologist.", ln=True)
    
    return pdf.output(dest='S').encode('latin-1')

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        model = joblib.load(MODEL_PATH)
        feature_names = joblib.load(FEATURES_PATH)
        metrics = joblib.load(METRICS_PATH)
        X_train_bg = joblib.load(BG_DATA_PATH)
        return model, feature_names, metrics, X_train_bg
    except FileNotFoundError:
        return None, None, None, None

# --- MAIN APP LOGIC ---
def main():
    inject_custom_css()
    init_db()
    model, feature_names, metrics, X_train_bg = load_assets()
    
    if not model:
        st.error("🚨 Critical Error: Model files not found.")
        st.info("Please run 'python train_model.py' first to generate the necessary files.")
        st.stop()

    # --- ADVANCED HEADER ---
    st.markdown("""
        <div class="gradient-header">
            <h1>LIV-AI Decision Support 🔬</h1>
            <p>Advanced Human-AI Collaborative Intelligence for Hepatic Risk Assessment</p>
        </div>
    """, unsafe_allow_html=True)

    # --- SIDEBAR INPUT ---
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>📋 Patient Vitals</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; opacity: 0.8;'>Enter biochemical markers below</p>", unsafe_allow_html=True)
        st.write("")
        
        with st.form("patient_form"):
            st.markdown("##### 👤 Demographics")
            c1, c2 = st.columns(2)
            age = c1.number_input("Age", 1, 100, 45)
            gender_txt = c2.selectbox("Gender", ["Male", "Female"])
            
            st.markdown("##### 🩸 Bilirubin Panel")
            tot_bil = st.number_input("Total Bilirubin (mg/dL)", 0.1, 50.0, 0.9, help="Normal: 0.1 - 1.2 mg/dL")
            dir_bil = st.number_input("Direct Bilirubin (mg/dL)", 0.1, 30.0, 0.2, help="Normal: < 0.3 mg/dL")
            
            st.markdown("##### 🧪 Hepatic Enzymes")
            alk_phos = st.number_input("Alkaline Phosphatase", 10, 2000, 200, help="Normal: 44 - 147 IU/L")
            alamine = st.number_input("ALT (IU/L)", 10, 2000, 20, help="Normal: 7 - 56 IU/L")
            aspartate = st.number_input("AST (IU/L)", 10, 2000, 20, help="Normal: 8 - 33 IU/L")
            
            st.markdown("##### 🧬 Proteins")
            tot_prot = st.number_input("Total Proteins (g/dL)", 1.0, 10.0, 6.8, help="Normal: 6.0 - 8.3 g/dL")
            albumin = st.number_input("Albumin (g/dL)", 0.5, 6.0, 3.3, help="Normal: 3.4 - 5.4 g/dL")
            ag_ratio = st.number_input("A/G Ratio", 0.1, 3.0, 0.9, help="Normal: 1.1 - 2.5")
            
            st.write("") 
            submit_btn = st.form_submit_button("✨ Execute AI Scan", use_container_width=True)

    # --- INPUT PROCESSING ---
    if submit_btn:
        input_dict = {
            'Age': age, 
            'Gender': 1 if gender_txt == "Male" else 0, 
            'Total_Bilirubin': tot_bil, 
            'Direct_Bilirubin': dir_bil, 
            'Alkaline_Phosphotase': alk_phos, 
            'Alamine_Aminotransferase': alamine, 
            'Aspartate_Aminotransferase': aspartate,
            'Total_Protiens': tot_prot, 
            'Albumin': albumin, 
            'Albumin_and_Globulin_Ratio': ag_ratio
        }
        input_df = pd.DataFrame([input_dict])
        
        probs = model.predict_proba(input_df)[0]
        pred = model.predict(input_df)[0]
        
        tree_preds = [tree.predict_proba(input_df.values)[0][1] for tree in model.estimators_]
        std_dev = np.std(tree_preds)
        
        st.session_state.update({
            'prediction_made': True, 
            'last_input': input_dict, 
            'last_input_df': input_df,
            'last_pred': pred, 
            'last_probs': probs, 
            'last_std': std_dev
        })

    # --- DASHBOARD TABS ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚀 Clinical Diagnosis", 
        "🧠 Multi-XAI Interpretability", 
        "🌍 Global Analytics", 
        "⚙️ HITL Retraining"
    ])

    # TAB 1: PREDICTION & UNCERTAINTY
    with tab1:
        if st.session_state['prediction_made']:
            input_dict = st.session_state['last_input']
            pred = st.session_state['last_pred']
            probs = st.session_state['last_probs']
            std_dev = st.session_state['last_std']
            
            target_prob = probs[1] if pred == 1 else probs[0]
            ci_lower = max(0.0, target_prob - (1.96 * std_dev))
            ci_upper = min(1.0, target_prob + (1.96 * std_dev))
            ci_text = f"[{ci_lower*100:.1f}% - {ci_upper*100:.1f}%]"
            
            if std_dev < 0.15: 
                rel_color, reliability = "#059669", "✨ 🟢 High Reliability"
            elif std_dev < 0.3: 
                rel_color, reliability = "#D97706", "✨ 🟡 Moderate Reliability"
            else: 
                rel_color, reliability = "#DC2626", "✨ 🔴 Low Reliability (Review)"

            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2 = st.columns([1.5, 1], gap="large")
            
            with col1:
                if pred == 1:
                    st.markdown(f"""
                        <div class="risk-card">
                            <h2 style="margin:0; font-size: 2.2rem; font-weight:800;">⚠️ HIGH RISK</h2>
                            <h4 style="margin:0; opacity:0.9;">Liver Disease Detected</h4>
                            <h1 style="font-size: 4rem; margin: 15px 0;">{target_prob*100:.1f}%</h1>
                            <p style="margin:0; font-size:1.1rem; opacity:0.9;">Pathogenic biomarker patterns identified. Clinical intervention recommended.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="safe-card">
                            <h2 style="margin:0; font-size: 2.2rem; font-weight:800;">✅ LOW RISK</h2>
                            <h4 style="margin:0; opacity:0.9;">Healthy Profile</h4>
                            <h1 style="font-size: 4rem; margin: 15px 0;">{target_prob*100:.1f}%</h1>
                            <p style="margin:0; font-size:1.1rem; opacity:0.9;">Biomarkers align with standard healthy ranges. No abnormalities detected.</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                st.progress(float(target_prob))
            
            with col2:
                st.markdown(f"""
                    <div class="info-card">
                        <h3 style="color: #FFFFFF; margin-top: 0; font-weight: 800;">Statistical Uncertainty 🎯</h3>
                        <p style="color: #64748B; margin-bottom: 5px;">95% Confidence Interval</p>
                        <h2 style="color: #0EA5E9; font-size: 2.2rem; margin-top:0;">{ci_text}</h2>
                        <hr style="border: 0; height: 1px; background: #E2E8F0; margin: 20px 0;">
                        <h3 style="color: {rel_color}; margin: 0;">{reliability}</h3>
                        <p style="font-size: 0.85rem; color: #94A3B8; margin-top:10px;">Ensemble variance calculated dynamically across 100 internal estimators.</p>
                    </div>
                """, unsafe_allow_html=True)
                
            st.markdown("<br><br>", unsafe_allow_html=True)
            
            # Action Buttons Row
            btn_col1, btn_col2, btn_col3 = st.columns(3)
            with btn_col1:
                pdf_bytes = create_pdf(input_dict, pred, f"{target_prob*100:.1f}%", ci_text, reliability)
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(f'<a href="data:application/octet-stream;base64,{b64}" download="Patient_Report.pdf" style="text-decoration: none;"><button style="width:100%; padding: 15px; background:linear-gradient(135deg, #0F172A, #334155); color:white; border:none; border-radius:12px; cursor:pointer; font-weight:bold; font-size:1rem; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">📄 Download Medical PDF</button></a>', unsafe_allow_html=True)
            
            with btn_col2:
                if st.button("✅ Confirm Prediction (Save)", use_container_width=True):
                    save_feedback(input_dict, pred, pred)
                    st.toast("Verification Saved successfully to database.", icon="✅")
            with btn_col3:
                if st.button("❌ Override Prediction (Retrain)", use_container_width=True):
                    correct_label = 0 if pred == 1 else 1
                    save_feedback(input_dict, pred, correct_label)
                    st.toast("Override Recorded. Logged for Active Learning.", icon="🔄")

        else:
            st.info("👈 Waiting for input... Please enter patient vitals in the sidebar and click **Execute AI Scan**.")

    # TAB 2: EXPLAINABILITY (XAI)
    with tab2:
        if st.session_state['prediction_made']:
            st.markdown("### 🧠 Transparent AI Reasoning")
            st.markdown("Dismantling the black box to provide mathematically exact clinical justifications.")
            st.write("")
            xai_method = st.radio("Select Mathematical Explainer Engine:", 
                                ["SHAP (Game Theory - Exact Attribution)", "LIME (Local Surrogate Perturbation)"], 
                                horizontal=True)
            st.write("")
            
            if "SHAP" in xai_method:
                st.success("💡 **Interpretation:** RED bars push the patient towards disease. BLUE bars pull the patient towards healthy.")
                with st.spinner("Calculating Shapley Values..."):
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(st.session_state['last_input_df'])
                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_values[0][:, 1], show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
                
            elif "LIME" in xai_method:
                st.info("💡 **Interpretation:** LIME builds a linear regression model precisely around this patient to identify threshold triggers.")
                with st.spinner("Compiling localized surrogate model..."):
                    explainer = LimeTabularExplainer(
                        X_train_bg.values, 
                        feature_names=X_train_bg.columns, 
                        class_names=['Healthy', 'Disease'], 
                        mode='classification'
                    )
                    exp = explainer.explain_instance(st.session_state['last_input_df'].iloc[0], model.predict_proba, num_features=8)
                    components.html(exp.as_html(), height=450, scrolling=True)
        else:
            st.warning("Generate a prediction first to unlock XAI features.")

    # TAB 3: GLOBAL ANALYTICS
    with tab3:
        st.markdown("### 🌍 Global Model Performance Analytics")
        st.markdown("Validation metrics based on the Indian Liver Patient Dataset (ILPD) Benchmarking.")
        
        if 'comparison_table' in metrics:
            comp_df = pd.DataFrame(metrics['comparison_table'])
            
            # FIX: Convert percentage strings (e.g., '73.5%') back to floats so Pandas can apply the color gradient
            metric_cols = [c for c in ['Accuracy', 'Precision', 'Recall', 'F1-Score'] if c in comp_df.columns]
            for col in metric_cols:
                comp_df[col] = comp_df[col].astype(str).str.replace('%', '').astype(float)
                
            # Apply styling and format back to strings with '%' symbol for display
            styled_df = comp_df.style.background_gradient(
                cmap='Blues', subset=['Accuracy', 'F1-Score']
            ).format({col: "{:.1f}%" for col in metric_cols})
            
            st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.write("")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("#### 🎯 Model Confusion Matrix")
            cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                               index=["Actual Healthy", "Actual Disease"], 
                               columns=["Pred Healthy", "Pred Disease"])
            st.dataframe(cm_df.style.background_gradient(cmap='Reds'), use_container_width=True)
            
        with col_m2:
            st.markdown("#### ⚖️ Global Feature Importance")
            fig, ax = plt.subplots(figsize=(6, 4))
            importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
            importances.tail(6).plot.barh(color='#0EA5E9', ax=ax, edgecolor='black')
            ax.set_xlabel("Relative Importance Weight", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)

    # TAB 4: ACTIVE LEARNING & DISAGREEMENT
    with tab4:
        st.markdown("### ⚙️ Human-in-the-Loop Active Learning")
        st.markdown("Review clinician disagreements and dynamically retrain the Random Forest to prevent concept drift.")
        
        conn = sqlite3.connect(DB_PATH)
        df_fb = pd.read_sql("SELECT * FROM liver_feedback", conn)
        conn.close()
        
        overrides = df_fb[df_fb['agreement'] == 'No'] if not df_fb.empty else pd.DataFrame()
        
        st.write("")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Total Clinical Scans", len(df_fb))
        metrics_col2.metric("Pending Disagreements (Overrides)", len(overrides))
        metrics_col3.metric("Baseline Model Accuracy", "73.5%")
        
        st.divider()
        
        col_r1, col_r2 = st.columns([2, 1])
        with col_r1:
            st.markdown("#### 🔄 Trigger Retraining Pipeline")
            st.write("Clicking this button will merge all verified human overrides with the original training matrix and completely re-compile the model's high-dimensional decision boundaries.")
        with col_r2:
            st.write("")
            if st.button("🚀 Execute Model Retraining", type="primary", use_container_width=True):
                if len(overrides) == 0:
                    st.info("No overrides available to warrant model retraining at this time.")
                else:
                    with st.spinner("Recompiling Random Forest Decision Boundaries..."):
                        try:
                            msg = retrain_from_feedback.retrain_system()
                            if "Success" in msg:
                                st.success(msg)
                                st.cache_resource.clear()
                                st.session_state['prediction_made'] = False 
                                st.rerun()
                            else:
                                st.error(msg)
                        except Exception as e:
                            st.error(f"Kernel Error during retraining: {e}")
        
        if len(overrides) > 0:
            st.markdown("#### 📄 Raw Override Log")
            st.dataframe(overrides[['Age', 'Gender', 'ai_prediction', 'human_label', 'timestamp']], use_container_width=True)

if __name__ == '__main__':
    main()