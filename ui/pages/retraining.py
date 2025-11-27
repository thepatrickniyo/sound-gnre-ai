"""
Retraining Page for Music Genre Classification

Allows users to retrain the model with new data.
"""
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import time
from datetime import datetime

from ui.utils.api_client import get_api_client

# Page configuration
st.set_page_config(
    page_title="Retraining",
    page_icon=":arrows_counterclockwise:",
    layout="wide"
)

st.title("Model Retraining")
st.markdown("Retrain the model with new data to improve performance.")

# Initialize API client
api_base_url = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:8000",
    help="Base URL of the FastAPI server"
)
api_client = get_api_client(api_base_url)

# Check API connection
health = api_client.health_check()
if not health:
    st.error("Cannot connect to API. Please ensure the API server is running at the specified URL.")
    st.stop()

# Get current model status
st.header("Current Model Information")

model_status = api_client.get_model_status()
if model_status:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Version", model_status.get('model_version', 'N/A'))
    with col2:
        st.metric("Model Type", model_status.get('model_type', 'N/A'))
    with col3:
        if model_status.get('test_accuracy'):
            st.metric("Test Accuracy", f"{model_status['test_accuracy']:.1%}")
        else:
            st.metric("Test Accuracy", "N/A")
    with col4:
        status_text = "Loaded" if model_status.get('is_loaded') else "Not Loaded"
        st.metric("Status", status_text)
    
    if model_status.get('training_date'):
        st.caption(f"Training Date: {model_status['training_date']}")
else:
    st.warning("⚠️ Could not retrieve model status.")

st.markdown("---")

# Retraining configuration
st.header("Retraining Configuration")

col1, col2 = st.columns(2)

with col1:
    model_version = st.selectbox(
        "Model Version to Retrain",
        options=["Latest (auto-detect)", "Specify version"],
        help="Select which model version to retrain"
    )
    
    if model_version == "Specify version":
        version_input = st.text_input(
            "Enter model version",
            placeholder="e.g., v1_20251126_211324",
            help="Enter the exact model version name"
        )
        selected_version = version_input if version_input else None
    else:
        selected_version = None
    
    combine_strategy = st.selectbox(
        "Data Combination Strategy",
        options=["append", "replace", "merge"],
        help="How to combine new data with existing data"
    )
    
    use_existing_data = st.checkbox(
        "Use Existing Training Data",
        value=True,
        help="Whether to include existing training data in retraining"
    )

with col2:
    new_version = st.text_input(
        "New Model Version Name",
        placeholder="e.g., v2_20251127_120000",
        help="Optional: Specify a name for the new model version. If not provided, a timestamp-based name will be generated."
    )
    
    st.info("""
    **Retraining Process:**
    1. Load existing model and data
    2. Load new data from uploads directory
    3. Combine datasets based on strategy
    4. Retrain model
    5. Evaluate and save new model
    
    **Note:** Retraining may take several minutes.
    """)

st.markdown("---")

# Start retraining
st.header("Start Retraining")

if st.button("Start Retraining", type="primary", use_container_width=True):
    if not model_status or not model_status.get('is_loaded'):
        st.error("❌ No model is currently loaded. Please train a model first.")
    else:
        with st.spinner("Starting retraining job..."):
            result = api_client.retrain_model(
                model_version=selected_version,
                combine_strategy=combine_strategy,
                use_existing_data=use_existing_data,
                new_version=new_version if new_version else None
            )
            
            if result:
                job_id = result.get('job_id')
                if job_id:
                    st.success("Retraining job started successfully!")
                    st.info(f"**Job ID:** `{job_id}`")
                    st.session_state['retraining_job_id'] = job_id
                    st.rerun()
                else:
                    st.error("Failed to start retraining job. No job ID returned.")
            else:
                st.error("Failed to start retraining job. Please check the API connection.")

# Training status monitoring
if 'retraining_job_id' in st.session_state:
    st.markdown("---")
    st.header("Training Status")
    
    job_id = st.session_state['retraining_job_id']
    
    # Get and display status
    status_result = api_client.get_training_status(job_id)
    
    if status_result:
        status = status_result.get('status', 'unknown')
        progress = status_result.get('progress', 0)
        message = status_result.get('message', '')
        result = status_result.get('result')
        
        # Display status
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Status", status.upper())
        with col2:
            st.progress(progress / 100 if progress else 0)
            st.caption(f"Progress: {progress:.1f}%")
        
        st.info(f"**Message:** {message}")
        
        # Show result if completed
        if status == 'completed' and result:
            st.success("Retraining completed successfully!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'new_version' in result:
                    st.metric("New Version", result['new_version'])
            with col2:
                if 'old_version' in result:
                    st.metric("Old Version", result['old_version'])
            with col3:
                if 'test_accuracy' in result:
                    st.metric("Test Accuracy", f"{result['test_accuracy']:.1%}")
            
            # Show comparison
            if 'old_version' in result and 'new_version' in result:
                st.subheader("Model Comparison")
                import pandas as pd
                comparison_data = {
                    'Version': [result.get('old_version', 'N/A'), result.get('new_version', 'N/A')],
                    'Test Accuracy': [
                        f"{result.get('old_accuracy', 0):.1%}" if 'old_accuracy' in result else "N/A",
                        f"{result.get('test_accuracy', 0):.1%}" if 'test_accuracy' in result else "N/A"
                    ]
                }
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        elif status == 'failed':
            st.error("Retraining failed!")
            error_msg = status_result.get('error', 'Unknown error')
            if error_msg:
                st.error(f"**Error:** {error_msg}")
        
        # Auto-refresh for pending/running jobs
        if status in ['pending', 'running']:
            auto_refresh = st.checkbox("Auto-refresh status", value=True, key=f"auto_refresh_{job_id}")
            if auto_refresh:
                time.sleep(2)
                st.rerun()
    else:
        st.error("Could not retrieve training status.")
    
    # Manual refresh button
    if st.button("Refresh Status", key=f"refresh_{job_id}"):
        st.rerun()

# All training jobs
st.markdown("---")
st.header("All Training Jobs")

if st.button("Refresh Job List"):
    all_jobs = api_client.get_all_training_status()
    
    if all_jobs:
        total_jobs = all_jobs.get('total_jobs', 0)
        st.metric("Total Jobs", total_jobs)
        
        jobs = all_jobs.get('jobs', {})
        if jobs:
            import pandas as pd
            
            jobs_data = []
            for job_id, job_info in jobs.items():
                jobs_data.append({
                    'Job ID': job_id[:8] + '...',  # Truncate for display
                    'Status': job_info.get('status', 'unknown'),
                    'Progress': f"{job_info.get('progress', 0):.1f}%",
                    'Message': job_info.get('message', 'N/A'),
                    'Created': job_info.get('created_at', 'N/A')
                })
            
            jobs_df = pd.DataFrame(jobs_data)
            st.dataframe(jobs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No training jobs found.")
    else:
        st.warning("Could not retrieve training jobs.")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.header("Information")
st.sidebar.info(
    """
    **Retraining Process:**
    
    1. Configure retraining parameters
    2. Click "Start Retraining"
    3. Monitor progress in real-time
    4. View results when complete
    
    **Data Combination Strategies:**
    - **append**: Add new data to existing data
    - **replace**: Use only new data
    - **merge**: Intelligently merge datasets
    
    **Note:** Ensure you have uploaded new training data before retraining.
    """
)

