"""
Upload Data Page for Music Genre Classification

Allows users to upload multiple audio files for training.
"""
import streamlit as st
from pathlib import Path
import time

from ui.utils.api_client import get_api_client
from ui.utils.audio_utils import validate_audio_file, get_audio_info, format_file_size

# Page configuration
st.set_page_config(
    page_title="Upload Data",
    page_icon=":folder:",
    layout="wide"
)

st.title("Upload Training Data")
st.markdown("Upload audio files to be used for model training or retraining.")

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

# File upload section
st.header("Upload Audio Files")

uploaded_files = st.file_uploader(
    "Choose audio files to upload",
    type=['wav', 'mp3', 'flac', 'm4a', 'ogg'],
    accept_multiple_files=True,
    help="You can upload multiple files at once. Supported formats: WAV, MP3, FLAC, M4A, OGG"
)

if uploaded_files:
    st.markdown("---")
    
    # Display uploaded files
    st.subheader("Uploaded Files")
    
    files_data = []
    valid_files = []
    invalid_files = []
    
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        
        # Validate file
        is_valid, error_msg = validate_audio_file(file_bytes, filename)
        file_info = get_audio_info(file_bytes, filename)
        
        file_data = {
            'filename': filename,
            'size': len(file_bytes),
            'valid': is_valid,
            'error': error_msg if not is_valid else None,
            'info': file_info
        }
        
        files_data.append(file_data)
        
        if is_valid:
            valid_files.append((filename, file_bytes))
        else:
            invalid_files.append((filename, error_msg))
    
    # Display file list
    if files_data:
        import pandas as pd
        
        display_data = []
        for fd in files_data:
            display_data.append({
                'Filename': fd['filename'],
                'Size': format_file_size(fd['size']),
                'Status': 'Valid' if fd['valid'] else 'Invalid',
                'Error': fd['error'] if fd['error'] else '-'
            })
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Files", len(files_data))
        with col2:
            st.metric("Valid Files", len(valid_files), delta=f"{len(valid_files) - len(invalid_files)}")
        with col3:
            st.metric("Invalid Files", len(invalid_files))
        
        # Show file details
        with st.expander("File Details", expanded=False):
            for fd in files_data:
                if fd['valid']:
                    st.write(f"**{fd['filename']}**")
                    info = fd['info']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"Size: {format_file_size(info['size_bytes'])}")
                    with col2:
                        if 'duration_seconds' in info:
                            st.caption(f"Duration: {info['duration_seconds']}s")
                    with col3:
                        if 'framerate' in info:
                            st.caption(f"Sample Rate: {info['framerate']} Hz")
                    st.markdown("---")
    
    # Upload button
    if valid_files:
        st.markdown("---")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            upload_button = st.button(
                "Upload to Server",
                type="primary",
                use_container_width=True
            )
        
        if upload_button:
            with st.spinner(f"Uploading {len(valid_files)} file(s)..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Upload files
                result = api_client.upload_data(valid_files)
                
                if result:
                    progress_bar.progress(100)
                    status_text.empty()
                    
                    uploaded_count = result.get('uploaded', 0)
                    failed_count = result.get('failed', 0)
                    
                    if uploaded_count > 0:
                        st.success(f"Successfully uploaded {uploaded_count} file(s)!")
                        
                        # Show uploaded files
                        uploaded_file_list = result.get('uploaded_files', [])
                        if uploaded_file_list:
                            st.subheader("Successfully Uploaded Files")
                            import pandas as pd
                            upload_df = pd.DataFrame(uploaded_file_list)
                            st.dataframe(upload_df, use_container_width=True, hide_index=True)
                    
                    if failed_count > 0:
                        st.warning(f"{failed_count} file(s) failed to upload.")
                        
                        # Show failed files
                        failed_file_list = result.get('failed_files', [])
                        if failed_file_list:
                            st.subheader("Failed Files")
                            import pandas as pd
                            failed_df = pd.DataFrame(failed_file_list)
                            st.dataframe(failed_df, use_container_width=True, hide_index=True)
                else:
                    st.error("Failed to upload files. Please check the API connection and try again.")
    
    # Show invalid files
    if invalid_files:
        st.markdown("---")
        st.warning("Some files are invalid and cannot be uploaded:")
        for filename, error in invalid_files:
            st.error(f"**{filename}**: {error}")

# Upload directory info
st.markdown("---")
st.header("Upload Information")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload Directory")
    uploads_dir = Path("data/uploads")
    if uploads_dir.exists():
        file_count = len(list(uploads_dir.glob("*")))
        st.info(f"Files are saved to: `{uploads_dir}`")
        st.metric("Files in upload directory", file_count)
    else:
        st.info(f"Upload directory will be created at: `{uploads_dir}`")

with col2:
    st.subheader("Guidelines")
    st.markdown("""
    **File Requirements:**
    - Supported formats: WAV, MP3, FLAC, M4A, OGG
    - Maximum file size: 50 MB per file
    - Recommended duration: 3-30 seconds
    
    **Best Practices:**
    - Use high-quality audio files
    - Ensure files are properly labeled
    - Upload files in batches for better organization
    """)

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.header("Information")
st.sidebar.info(
    """
    **How to use:**
    1. Select one or more audio files
    2. Review the file list and validation status
    3. Click "Upload to Server" to upload valid files
    4. Files will be saved to the uploads directory
    
    **Note:** Uploaded files can be used for model retraining.
    """
)

