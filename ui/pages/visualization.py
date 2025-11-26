"""
Visualization Page for Music Genre Classification

Displays feature visualizations and dataset insights.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Visualizations",
    page_icon=":bar_chart:",
    layout="wide"
)

st.title("Data & Feature Visualizations")
st.markdown("Explore the dataset features and model performance metrics.")

# Load dataset
@st.cache_data
def load_dataset():
    """Load the feature dataset."""
    try:
        # Try 30-second features first
        csv_path = Path("data/dataset/features_30_sec.csv")
        if not csv_path.exists():
            csv_path = Path("data/dataset/features_3_sec.csv")
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return df
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Load dataset
df = load_dataset()

if df is None:
    st.error("Dataset not found. Please ensure the dataset files are in the correct location.")
    st.info("Expected location: `data/dataset/features_30_sec.csv` or `data/dataset/features_3_sec.csv`")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
genres = sorted(df['label'].unique()) if 'label' in df.columns else []
selected_genres = st.sidebar.multiselect(
    "Select genres to display",
    genres,
    default=genres[:5] if len(genres) >= 5 else genres
)

if not selected_genres:
    st.warning("Please select at least one genre to visualize.")
    st.stop()

# Filter dataframe
df_filtered = df[df['label'].isin(selected_genres)] if selected_genres else df

st.markdown("---")

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Feature 1: Tempo Distribution",
    "Feature 2: MFCC Coefficients",
    "Feature 3: Spectral Centroid",
    "Dataset Overview"
])

# Tab 1: Tempo Distribution
with tab1:
    st.header("Tempo Distribution Across Genres")
    st.markdown("""
    **Story:** Different music genres have distinct tempo ranges. 
    Tempo (beats per minute) is a fundamental characteristic that helps distinguish genres.
    For example, classical music often has variable tempos, while disco typically has 
    a consistent, upbeat tempo around 120-130 BPM.
    """)
    
    if 'tempo' in df_filtered.columns:
        # Box plot
        fig_box = px.box(
            df_filtered,
            x='label',
            y='tempo',
            color='label',
            title="Tempo Distribution by Genre",
            labels={'label': 'Genre', 'tempo': 'Tempo (BPM)'}
        )
        fig_box.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Violin plot
        fig_violin = px.violin(
            df_filtered,
            x='label',
            y='tempo',
            color='label',
            title="Tempo Distribution (Violin Plot)",
            labels={'label': 'Genre', 'tempo': 'Tempo (BPM)'}
        )
        fig_violin.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_violin, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Tempo Statistics by Genre")
            tempo_stats = df_filtered.groupby('label')['tempo'].agg(['mean', 'std', 'min', 'max']).round(2)
            st.dataframe(tempo_stats, use_container_width=True)
        
        with col2:
            st.subheader("Insights")
            avg_tempo = df_filtered.groupby('label')['tempo'].mean().sort_values(ascending=False)
            st.write("**Average Tempo (BPM) by Genre:**")
            for genre, tempo in avg_tempo.items():
                st.write(f"- **{genre.capitalize()}**: {tempo:.1f} BPM")
    else:
        st.warning("Tempo feature not found in dataset.")

# Tab 2: MFCC Coefficients
with tab2:
    st.header("MFCC Coefficients Comparison")
    st.markdown("""
    **Story:** MFCC (Mel-Frequency Cepstral Coefficients) features capture timbral characteristics 
    unique to each genre. These coefficients represent the spectral shape of the audio and are 
    crucial for distinguishing between genres with similar tempos but different timbres.
    """)
    
    # Find MFCC columns
    mfcc_cols = [col for col in df_filtered.columns if 'mfcc' in col.lower()]
    
    if mfcc_cols:
        # Select which MFCC coefficients to visualize
        selected_mfcc = st.multiselect(
            "Select MFCC coefficients to visualize",
            mfcc_cols,
            default=mfcc_cols[:5] if len(mfcc_cols) >= 5 else mfcc_cols
        )
        
        if selected_mfcc:
            # Heatmap of average MFCC values by genre
            mfcc_avg = df_filtered.groupby('label')[selected_mfcc].mean()
            
            fig_heatmap = px.imshow(
                mfcc_avg.T,
                labels=dict(x="Genre", y="MFCC Coefficient", color="Average Value"),
                title="Average MFCC Coefficients by Genre (Heatmap)",
                aspect="auto",
                color_continuous_scale="Viridis"
            )
            fig_heatmap.update_layout(height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Line plot comparing genres
            fig_line = go.Figure()
            
            for genre in selected_genres:
                genre_data = df_filtered[df_filtered['label'] == genre]
                avg_mfcc = genre_data[selected_mfcc].mean()
                
                fig_line.add_trace(go.Scatter(
                    x=selected_mfcc,
                    y=avg_mfcc,
                    mode='lines+markers',
                    name=genre.capitalize()
                ))
            
            fig_line.update_layout(
                title="Average MFCC Coefficients by Genre",
                xaxis_title="MFCC Coefficient",
                yaxis_title="Average Value",
                height=500
            )
            st.plotly_chart(fig_line, use_container_width=True)
            
            # Statistics table
            st.subheader("MFCC Statistics by Genre")
            mfcc_stats = df_filtered.groupby('label')[selected_mfcc].agg(['mean', 'std']).round(3)
            st.dataframe(mfcc_stats, use_container_width=True)
    else:
        st.warning("MFCC features not found in dataset.")

# Tab 3: Spectral Centroid
with tab3:
    st.header("Spectral Centroid Distribution")
    st.markdown("""
    **Story:** Spectral centroid indicates the "brightness" of a sound. Higher spectral centroid 
    values indicate brighter/harsher sounds (like metal), while lower values indicate 
    darker/mellower sounds (like blues or jazz). This feature helps distinguish genres based 
    on their tonal characteristics.
    """)
    
    if 'spectral_centroid_mean' in df_filtered.columns:
        # Histogram
        fig_hist = px.histogram(
            df_filtered,
            x='spectral_centroid_mean',
            color='label',
            nbins=50,
            title="Spectral Centroid Distribution by Genre",
            labels={'spectral_centroid_mean': 'Spectral Centroid', 'count': 'Frequency'},
            barmode='overlay',
            opacity=0.7
        )
        fig_hist.update_layout(height=500)
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Density plot (KDE)
        fig_density = go.Figure()
        
        for genre in selected_genres:
            genre_data = df_filtered[df_filtered['label'] == genre]
            fig_density.add_trace(go.Histogram(
                x=genre_data['spectral_centroid_mean'],
                name=genre.capitalize(),
                opacity=0.6,
                histnorm='probability density'
            ))
        
        fig_density.update_layout(
            title="Spectral Centroid Density by Genre",
            xaxis_title="Spectral Centroid",
            yaxis_title="Density",
            barmode='overlay',
            height=500
        )
        st.plotly_chart(fig_density, use_container_width=True)
        
        # Box plot
        fig_box = px.box(
            df_filtered,
            x='label',
            y='spectral_centroid_mean',
            color='label',
            title="Spectral Centroid Distribution (Box Plot)",
            labels={'label': 'Genre', 'spectral_centroid_mean': 'Spectral Centroid'}
        )
        fig_box.update_layout(height=500, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Spectral Centroid Statistics")
            centroid_stats = df_filtered.groupby('label')['spectral_centroid_mean'].agg(['mean', 'std', 'min', 'max']).round(3)
            st.dataframe(centroid_stats, use_container_width=True)
        
        with col2:
            st.subheader("Insights")
            avg_centroid = df_filtered.groupby('label')['spectral_centroid_mean'].mean().sort_values(ascending=False)
            st.write("**Average Spectral Centroid by Genre:**")
            for genre, centroid in avg_centroid.items():
                brightness = "Bright" if centroid > avg_centroid.median() else "Dark"
                st.write(f"- **{genre.capitalize()}**: {centroid:.2f} ({brightness})")
    else:
        st.warning("Spectral centroid feature not found in dataset.")

# Tab 4: Dataset Overview
with tab4:
    st.header("Dataset Overview")
    
    # Genre distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Genre Distribution")
        genre_counts = df_filtered['label'].value_counts()
        
        fig_pie = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title="Genre Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Sample Count by Genre")
        fig_bar = px.bar(
            x=genre_counts.index,
            y=genre_counts.values,
            labels={'x': 'Genre', 'y': 'Number of Samples'},
            title="Number of Samples per Genre",
            color=genre_counts.values,
            color_continuous_scale="Blues"
        )
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    st.markdown("---")
    
    # Feature correlation matrix
    st.subheader("Feature Correlation Matrix")
    
    # Select numeric features for correlation
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if 'length' in numeric_cols:
        numeric_cols.remove('length')
    
    if len(numeric_cols) > 1:
        # Sample features for correlation (to avoid performance issues)
        if len(numeric_cols) > 20:
            selected_features = st.multiselect(
                "Select features for correlation matrix",
                numeric_cols,
                default=numeric_cols[:15]
            )
        else:
            selected_features = numeric_cols
        
        if selected_features:
            corr_matrix = df_filtered[selected_features].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                labels=dict(x="Feature", y="Feature", color="Correlation"),
                title="Feature Correlation Matrix",
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # Dataset statistics
    st.markdown("---")
    st.subheader("Dataset Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Samples", len(df_filtered))
        st.metric("Number of Genres", df_filtered['label'].nunique())
        st.metric("Number of Features", len(df_filtered.columns) - 1)  # Exclude label
    
    with col2:
        st.write("**Dataset Info:**")
        st.write(f"- Shape: {df_filtered.shape[0]} rows × {df_filtered.shape[1]} columns")
        st.write(f"- Memory usage: {df_filtered.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        st.write(f"- Missing values: {df_filtered.isnull().sum().sum()}")

# Sidebar information
st.sidebar.markdown("---")
st.sidebar.header("Information")
st.sidebar.info(
    """
    **Visualization Features:**
    
    1. **Tempo Distribution**: Shows how different genres vary in tempo
    2. **MFCC Coefficients**: Displays timbral characteristics
    3. **Spectral Centroid**: Shows brightness/darkness of sounds
    4. **Dataset Overview**: Provides overall dataset statistics
    
    Use the filters to focus on specific genres.
    """
)

