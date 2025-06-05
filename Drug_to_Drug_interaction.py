
import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple
import time
from concurrent.futures import ThreadPoolExecutor
import requests
from drugData import DDIPredictor

st.set_page_config(
    page_title="DDI Predictor | Drug Interaction Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .drug-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
    
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
    }
    
    .warning-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ffc107;
        background-color: #fff3cd;
    }
    
    .danger-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
        background-color: #f8d7da;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if 'prediction_cache' not in st.session_state:
    st.session_state.prediction_cache = {}
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

@st.cache_resource
def load_model():
    """Load the DDI predictor model - cached across sessions"""
    try:
        predictor = DDIPredictor()
        

        model_dir = "models"
        if os.path.exists(model_dir):
            model_files = [f for f in os.listdir(model_dir) if f.startswith("ddi_model_")]
            if model_files:
                latest_model = max(model_files)
                timestamp = latest_model.replace("ddi_model_", "").replace(".pt", "")
                predictor.load_model(timestamp)
                logger.info(f"Model loaded successfully with timestamp: {timestamp}")
                return predictor, timestamp
        
      
        predictor.load_model()
        logger.info("Default model loaded successfully")
        return predictor, "default"
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        return None, None

@st.cache_data
def get_cached_prediction(drug1: str, drug2: str) -> Dict:
    """Get cached prediction result"""
    cache_key = f"{drug1.lower()}_{drug2.lower()}"
    reverse_key = f"{drug2.lower()}_{drug1.lower()}"
    
    if cache_key in st.session_state.prediction_cache:
        return st.session_state.prediction_cache[cache_key]
    elif reverse_key in st.session_state.prediction_cache:
        return st.session_state.prediction_cache[reverse_key]
    
    return None

def cache_prediction(drug1: str, drug2: str, result: Dict):
    """Cache prediction result"""
    cache_key = f"{drug1.lower()}_{drug2.lower()}"
    st.session_state.prediction_cache[cache_key] = result

def make_prediction(drug1: str, drug2: str) -> Dict:
    """Make prediction using the loaded model"""
    try:
        if st.session_state.predictor is None:
            return {"error": "Model not loaded"}
        
     
        cached_result = get_cached_prediction(drug1, drug2)
        if cached_result:
            return cached_result
        
       
        result = st.session_state.predictor.predict(drug1, drug2, return_details=True)
        
     
        if isinstance(result, dict) and 'error' not in result:
            cache_prediction(drug1, drug2, result)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return {"error": str(e)}

def display_prediction_result(result: Dict, drug1: str, drug2: str):
    """Display the prediction result in a formatted way"""
    if 'error' in result:
        st.error(f"❌ Error: {result['error']}")
        return
   
    if result.get('interaction_predicted', False):
        confidence = result.get('confidence', 'unknown')
        if confidence == 'high':
            card_class = "danger-card"
            icon = "🚨"
            status = "HIGH RISK INTERACTION DETECTED"
        else:
            card_class = "warning-card"
            icon = "⚠️"
            status = "POTENTIAL INTERACTION DETECTED"
    else:
        card_class = "result-card"
        icon = "✅"
        status = "NO SIGNIFICANT INTERACTION PREDICTED"
    

    st.markdown(f"""
    <div class="{card_class}">
        <h3>{icon} {status}</h3>
        <p><strong>Analysis Complete:</strong> {drug1} + {drug2}</p>
    </div>
    """, unsafe_allow_html=True)
    

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Interaction Probability",
            f"{result.get('probability', 0) * 100:.1f}%",
            delta=None
        )
    
    with col2:
        confidence_level = result.get('confidence', 'unknown').title()
        st.metric(
            "Confidence Level",
            confidence_level,
            delta=None
        )
    
    with col3:
        st.metric(
            "Drug 1",
            result.get('drug1_name', drug1),
            delta=None
        )
    
    with col4:
        st.metric(
            "Drug 2",
            result.get('drug2_name', drug2),
            delta=None
        )
    

    with st.expander("🔍 Detailed Drug Information", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Drug 1 Details")
            st.write(f"**Name:** {result.get('drug1_name', drug1)}")
            if 'drug1_formula' in result:
                st.write(f"**Formula:** {result['drug1_formula']}")
        
        with col2:
            st.markdown("### Drug 2 Details")
            st.write(f"**Name:** {result.get('drug2_name', drug2)}")
            if 'drug2_formula' in result:
                st.write(f"**Formula:** {result['drug2_formula']}")
    

    with st.expander("📋 Clinical Recommendations", expanded=True):
        if result.get('interaction_predicted', False):
            st.warning("""
            **⚠️ IMPORTANT MEDICAL ADVICE:**
            - Potential drug interaction detected
            - Consult with a healthcare provider immediately
            - Do not start, stop, or change medications without medical supervision
            - Monitor for adverse effects if already taking both medications
            """)
        else:
            st.success("""
            **✅ PRELIMINARY ASSESSMENT:**
            - No significant interaction predicted by the model
            - Always consult with a healthcare provider before combining medications
            - Individual patient factors may still affect drug interactions
            - Monitor for any unexpected side effects
            """)

def batch_prediction_interface():
    """Interface for batch predictions"""
    st.markdown("### 📊 Batch Prediction")
    
 
    uploaded_file = st.file_uploader(
        "Upload CSV file with drug pairs",
        type=['csv'],
        help="CSV should have columns: drug1, drug2"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            if 'drug1' in df.columns and 'drug2' in df.columns:
                st.write("**Preview of uploaded data:**")
                st.dataframe(df.head())
                
                if st.button("🚀 Run Batch Prediction"):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, row in df.iterrows():
                        result = make_prediction(row['drug1'], row['drug2'])
                        results.append({
                            'drug1': row['drug1'],
                            'drug2': row['drug2'],
                            'interaction_predicted': result.get('interaction_predicted', False),
                            'probability': result.get('probability', 0),
                            'confidence': result.get('confidence', 'unknown')
                        })
                        progress_bar.progress((idx + 1) / len(df))
                    
             
                    results_df = pd.DataFrame(results)
                    st.write("**Batch Prediction Results:**")
                    st.dataframe(results_df)
                    
  
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Pairs", len(results_df))
                    with col2:
                        interactions = results_df['interaction_predicted'].sum()
                        st.metric("Interactions Found", interactions)
                    with col3:
                        avg_prob = results_df['probability'].mean()
                        st.metric("Avg Probability", f"{avg_prob:.2f}")
                    

                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "batch_predictions.csv",
                        "text/csv"
                    )
            else:
                st.error("CSV must contain 'drug1' and 'drug2' columns")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def main():

    st.markdown("""
    <div class="main-header">
        <h1>💊 Drug Interaction Predictor</h1>
        <p>Advanced AI-powered system to predict potential interactions between medications</p>
    </div>
    """, unsafe_allow_html=True)
    

    if not st.session_state.model_loaded:
        with st.spinner("🔄 Loading AI model..."):
            predictor, timestamp = load_model()
            if predictor:
                st.session_state.predictor = predictor
                st.session_state.model_loaded = True
                st.success(f"✅ Model loaded successfully! (Version: {timestamp})")
            else:
                st.error("❌ Failed to load model. Please check your model files.")
                st.stop()
    

    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        

        st.info("**Model Status:** ✅ Loaded")
        

        cache_size = len(st.session_state.prediction_cache)
        st.metric("Cache Size", cache_size)
        
        if st.button("🗑️ Clear Cache"):
            st.session_state.prediction_cache.clear()
            st.success("Cache cleared!")
        
 
        st.markdown("### ⚙️ Settings")
        show_details = st.checkbox("Show detailed results", value=True)
        auto_predict = st.checkbox("Auto-predict on input", value=False)
        

        with st.expander("❓ Help & Info"):
            st.markdown("""
            **How to use:**
            1. Enter two drug names
            2. Click 'Predict Interaction'
            3. Review the results and recommendations
            
            **Features:**
            - Real-time predictions
            - Batch processing
            - Result caching
            - Detailed drug information
            """)
    

    tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📊 Batch Prediction", "📈 Analytics"])
    
    with tab1:
        st.markdown("### Enter Drug Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            drug1 = st.text_input(
                "First Drug 💊",
                placeholder="e.g., CHEMBL25, Aspirin",
                help="Enter drug name or ChEMBL ID"
            )
        
        with col2:
            drug2 = st.text_input(
                "Second Drug 💊",
                placeholder="e.g., CHEMBL2, Warfarin",
                help="Enter drug name or ChEMBL ID"
            )
        

        predict_button = st.button("🔬 Predict Interaction", type="primary", use_container_width=True)
        
        if (predict_button or (auto_predict and drug1 and drug2)) and drug1 and drug2:
            with st.spinner("🧠 Analyzing drug interaction..."):
                result = make_prediction(drug1, drug2)
                display_prediction_result(result, drug1, drug2)
        
        elif predict_button and (not drug1 or not drug2):
            st.warning("⚠️ Please enter both drug names")
    
    with tab2:
        batch_prediction_interface()
    
    with tab3:
        st.markdown("### 📈 Prediction Analytics")
        
        if st.session_state.prediction_cache:

            cache_data = []
            for key, result in st.session_state.prediction_cache.items():
                drugs = key.split('_')
                if len(drugs) == 2 and 'error' not in result:
                    cache_data.append({
                        'drug_pair': f"{drugs[0]} + {drugs[1]}",
                        'interaction_predicted': result.get('interaction_predicted', False),
                        'probability': result.get('probability', 0),
                        'confidence': result.get('confidence', 'unknown')
                    })
            
            if cache_data:
                cache_df = pd.DataFrame(cache_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Interaction Distribution")
                    interaction_counts = cache_df['interaction_predicted'].value_counts()
                    st.bar_chart(interaction_counts)
                
                with col2:
                    st.markdown("#### Probability Distribution")
                    fig_data = cache_df['probability'].value_counts().sort_index()
                    st.bar_chart(fig_data)
                
                st.markdown("#### Recent Predictions")
                st.dataframe(cache_df, use_container_width=True)
            else:
                st.info("No valid predictions in cache yet.")
        else:
            st.info("No predictions cached yet. Make some predictions to see analytics!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>⚕️ <strong>Medical Disclaimer:</strong> This tool is for research purposes only and should not replace professional medical advice.</p>
        <p>Always consult with qualified healthcare providers before making medication decisions.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()