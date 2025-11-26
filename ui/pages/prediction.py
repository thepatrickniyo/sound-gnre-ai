"""
Prediction Page for Music Genre Classification

Allows users to upload audio files or record audio and get genre predictions.
"""
import streamlit as st
import io
import time
import plotly.express as px
import plotly.graph_objects as go

from ui.utils.api_client import get_api_client
from ui.utils.audio_utils import validate_audio_file, get_audio_info

# Page configuration
st.set_page_config(
    page_title="Predict Genre",
    page_icon=":microphone:",
    layout="wide"
)

st.title("Music Genre Prediction")
st.markdown("Upload an audio file to predict its genre.")

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

# File upload
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
    help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
)

audio_data = None
filename = None

if uploaded_file is not None:
    audio_data = uploaded_file.read()
    filename = uploaded_file.name
    
    # Validate file
    is_valid, error_msg = validate_audio_file(audio_data, filename)
    if not is_valid:
        st.error(error_msg)
        st.stop()
    
    # Show file info
    file_info = get_audio_info(audio_data, filename)
    with st.expander("File Information", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Size", f"{file_info.get('size_mb', 0)} MB")
        with col2:
            if 'duration_seconds' in file_info:
                st.metric("Duration", f"{file_info['duration_seconds']}s")
        with col3:
            if 'framerate' in file_info:
                st.metric("Sample Rate", f"{file_info['framerate']} Hz")
    
    # Audio player
    st.audio(audio_data, format='audio/wav')

# Prediction section
if audio_data is not None:
    st.markdown("---")
    
    if st.button("Predict Genre", type="primary", use_container_width=True):
        with st.spinner("Processing audio and making prediction..."):
            start_time = time.time()
            
            # Make prediction
            result = api_client.predict(audio_data, filename)
            
            if result:
                processing_time = time.time() - start_time
                
                # Display results
                st.success("Prediction completed!")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Prediction")
                    predicted_genre = result.get('genre', 'Unknown')
                    confidence = result.get('confidence', 0)
                    
                    st.markdown(f"### {predicted_genre.capitalize()}")
                    st.metric("Confidence", f"{confidence:.1%}")
                    st.caption(f"Processing time: {result.get('processing_time_ms', 0):.0f} ms")
                
                with col2:
                    st.subheader("All Genre Probabilities")
                    
                    # Get all predictions
                    all_predictions = result.get('all_predictions', {})
                    
                    if all_predictions:
                        # Sort by confidence
                        sorted_predictions = sorted(
                            all_predictions.items(),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        
                        # Create bar chart
                        genres = [g.capitalize() for g, _ in sorted_predictions]
                        confidences = [c for _, c in sorted_predictions]
                        
                        fig = px.bar(
                            x=genres,
                            y=confidences,
                            labels={'x': 'Genre', 'y': 'Confidence'},
                            title="Genre Prediction Probabilities",
                            color=confidences,
                            color_continuous_scale="Blues"
                        )
                        fig.update_layout(
                            showlegend=False,
                            height=400,
                            yaxis=dict(range=[0, 1])
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show all predictions in a table
                        with st.expander("View All Predictions", expanded=False):
                            import pandas as pd
                            df = pd.DataFrame({
                                'Genre': [g.capitalize() for g, _ in sorted_predictions],
                                'Confidence': [f"{c:.1%}" for _, c in sorted_predictions]
                            })
                            st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Pie chart visualization
                st.subheader("Genre Distribution")
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    fig_pie = px.pie(
                        values=confidences,
                        names=genres,
                        title="Genre Probability Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    st.markdown("### Top 3 Predictions")
                    for i, (genre, conf) in enumerate(sorted_predictions[:3], 1):
                        st.markdown(f"**{i}. {genre.capitalize()}**")
                        st.progress(conf)
                        st.caption(f"{conf:.1%}")
            else:
                st.error("Failed to make prediction. Please check the API connection and try again.")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.header("Information")
st.sidebar.info(
    """
    **How to use:**
    1. Upload an audio file
    2. Click "Predict Genre" button
    3. View the predicted genre and confidence scores
    
    **Supported formats:**
    - WAV, MP3, FLAC, M4A, OGG
    
    **Note:** The model works best with audio clips of 3-30 seconds.
    """
)

# Model status in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Model Status")
model_status = api_client.get_model_status()
if model_status:
    if model_status.get('is_loaded'):
        st.sidebar.success("Model Loaded")
        st.sidebar.caption(f"Version: {model_status.get('model_version', 'N/A')}")
        st.sidebar.caption(f"Type: {model_status.get('model_type', 'N/A')}")
    else:
        st.sidebar.warning("Model Not Loaded")

