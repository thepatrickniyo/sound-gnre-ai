"""
Main Streamlit Application for Music Genre Classification

This is the main entry point for the Streamlit UI.
"""
import streamlit as st
from datetime import datetime
import time

from ui.utils.api_client import get_api_client

# Page configuration
st.set_page_config(
    page_title="Music Genre Classification",
    page_icon=":musical_note:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-healthy {
        color: #28a745;
        font-weight: bold;
    }
    .status-unhealthy {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


def main():
    """Main dashboard/home page."""
    st.title("Music Genre Classification")
    st.markdown("---")
    
    # Initialize API client
    api_base_url = st.sidebar.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="Base URL of the FastAPI server"
    )
    api_client = get_api_client(api_base_url)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    st.sidebar.markdown("""
    **Pages:**
    - Dashboard (current)
    - [Predict](?page=prediction)
    - [Visualizations](?page=visualization)
    - [Upload Data](?page=upload_data)
    - [Retraining](?page=retraining)
    """)
    
    # Note: In Streamlit multi-page apps, pages in the pages/ directory
    # are automatically available in the sidebar. This is the dashboard/home page.
    
    # Dashboard content
    st.header("Dashboard")
    
    # Health check
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("API Status")
        health = api_client.health_check()
        if health:
            st.markdown('<p class="status-healthy">Healthy</p>', unsafe_allow_html=True)
            st.caption(f"Version: {health.get('version', 'N/A')}")
            st.caption(f"Time: {health.get('timestamp', 'N/A')}")
        else:
            st.markdown('<p class="status-unhealthy">Unavailable</p>', unsafe_allow_html=True)
            st.error("Cannot connect to API. Please ensure the API server is running.")
    
    with col2:
        st.subheader("Model Status")
        model_status = api_client.get_model_status()
        if model_status:
            if model_status.get('is_loaded'):
                st.markdown('<p class="status-healthy">Loaded</p>', unsafe_allow_html=True)
                st.caption(f"Version: {model_status.get('model_version', 'N/A')}")
                st.caption(f"Type: {model_status.get('model_type', 'N/A')}")
                if model_status.get('test_accuracy'):
                    st.metric("Accuracy", f"{model_status['test_accuracy']:.1%}")
            else:
                st.markdown('<p class="status-unhealthy">Not Loaded</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-unhealthy">Unknown</p>', unsafe_allow_html=True)
    
    with col3:
        st.subheader("Uptime")
        stats = api_client.get_stats()
        if stats:
            uptime_hours = stats.get('uptime_hours', 0)
            st.metric("Uptime", f"{uptime_hours:.1f} hours")
            st.caption(f"Started: {stats.get('start_time', 'N/A')}")
        else:
            st.info("No statistics available")
    
    st.markdown("---")
    
    # Quick stats
    st.header("Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    stats = api_client.get_stats()
    if stats:
        with col1:
            st.metric("Total Requests", stats.get('total_requests', 0))
        with col2:
            st.metric("Predictions", stats.get('predictions', 0))
        with col3:
            st.metric("Training Jobs", stats.get('training_jobs', 0))
        with col4:
            st.metric("Errors", stats.get('errors', 0))
        
        if stats.get('uptime_hours', 0) > 0:
            rph = stats.get('requests_per_hour', 0)
            st.caption(f"Requests per hour: {rph:.1f}")
    else:
        st.info("No statistics available. Start using the API to see statistics.")
    
    st.markdown("---")
    
    # Model metrics
    st.header("Model Performance")
    
    metrics = api_client.get_metrics()
    if metrics:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
        with col2:
            st.metric("Precision", f"{metrics.get('precision', 0):.1%}")
        with col3:
            st.metric("Recall", f"{metrics.get('recall', 0):.1%}")
        with col4:
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.1%}")
        
        # Per-class metrics
        per_class = metrics.get('per_class_metrics')
        if per_class:
            st.subheader("Per-Genre Performance")
            import pandas as pd
            
            metrics_data = []
            for genre, genre_metrics in per_class.items():
                metrics_data.append({
                    'Genre': genre.capitalize(),
                    'Precision': f"{genre_metrics.get('precision', 0):.1%}",
                    'Recall': f"{genre_metrics.get('recall', 0):.1%}",
                    'F1-Score': f"{genre_metrics.get('f1', 0):.1%}"
                })
            
            df = pd.DataFrame(metrics_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No model metrics available. Train a model first.")
    
    st.markdown("---")
    
    # Quick links
    st.header("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("[Make Prediction](?page=prediction)")
    
    with col2:
        st.markdown("[Upload Data](?page=upload_data)")
    
    with col3:
        st.markdown("[Retrain Model](?page=retraining)")
    
    # Footer
    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

