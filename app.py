import os
import streamlit as st
import nest_asyncio
import pandas as pd
import numpy as np
import base64
import nltk
from model_adaptor import ModelAdaptor

# Apply nest_asyncio to handle async operations
nest_asyncio.apply()

# Download NLTK data at startup
try:
    nltk.download('punkt', quiet=True)
except:
    st.warning("Failed to download NLTK data. Some functionality might be limited.")

# Import from local modules for plotting only
from plot import (
    plot_similarity_matrix, plot_sentiment_timeline,
    plot_similarity_scores
)

# Set page configuration
st.set_page_config(
    page_title="Quantitative Semantic Analysis Python",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Define a custom theme
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_adaptor():
    """Load and cache the ModelAdaptor"""
    # Force CPU usage to avoid MPS issues
    adaptor = ModelAdaptor(force_cpu=True)
    return adaptor


def segment_text(text, method, segment_size):
    """
    Segment the text based on specified method and size with robust
    fallbacks for when NLTK isn't available.
    """
    import re

    # Clean the text first
    text = re.sub(r'\s+', ' ', text).strip()

    if method == 'sentences':
        # First try using nltk with proper error handling
        try:
            import nltk

            # Try to ensure punkt is downloaded (normal version)
            try:
                nltk.download('punkt', quiet=True)
            except:
                pass

            try:
                # Try standard punkt tokenizer
                from nltk.tokenize import sent_tokenize
                sentences = sent_tokenize(text)
            except (ImportError, LookupError):
                # Fall back to simple regex-based sentence splitting
                sentences = re.split(r'(?<=[.!?])\s+', text)
        except:
            # If NLTK completely fails, use regex fallback
            sentences = re.split(r'(?<=[.!?])\s+', text)

        # Create segments from sentences
        segments = []
        for i in range(0, len(sentences), segment_size):
            segment = ' '.join(sentences[i:i + segment_size])
            segments.append(segment)

    elif method == 'words':
        # Segment by words - doesn't require NLTK
        words = text.split()
        segments = []

        for i in range(0, len(words), segment_size):
            segment = ' '.join(words[i:i + segment_size])
            segments.append(segment)

    elif method == 'paragraphs':
        # Segment by paragraphs - doesn't require NLTK
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        segments = []

        for i in range(0, len(paragraphs), segment_size):
            segment = ' '.join(paragraphs[i:i + segment_size])
            segments.append(segment)

    # Filter out empty segments
    segments = [s for s in segments if s.strip()]

    # Ensure there's at least one segment
    if not segments:
        segments = [text]

    return segments


def get_download_link(df, filename="interview_analysis.csv", text="Download CSV"):
    """Generate a download link for the DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href


def main():
    # App title
    st.title("QSP")
    st.markdown("""
    This tool analyzes interview text to extract sentiment and context patterns 
    for further modeling with HMM (Hidden Markov Models).
    """)

    # Load the model adaptor
    adaptor = load_model_adaptor()

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Text segmentation options
    st.sidebar.subheader("Text Segmentation")
    segmentation_method = st.sidebar.selectbox(
        "Segmentation Method",
        options=["sentences", "words", "paragraphs"],
        index=0,
        help="How to divide the text into segments for analysis"
    )

    segment_size = st.sidebar.slider(
        "Segment Size",
        min_value=1,
        max_value=20,
        value=3,
        help="Number of units (sentences, words, or paragraphs) per segment"
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "QSP- Quantitative semantic analysis Python\n\n"
        "MIT License\n\n"
        "Author: Alexander Engelmark Linköping University\n\n"
        "Please cite https://github.com/LazyCyborg/QSP if using this software.\n\n "
    )

    # Main content
    tab1, tab2, tab3 = st.tabs(["Text Input", "Analysis Results", "Visualizations"])

    with tab1:
        st.header("Input Interview Text")

        # Sample text checkbox
        use_sample = st.checkbox("Use Sample Text")

        # Sample text
        sample_text = """
        I've been reflecting on my journey in this company. When I started five years ago, I was excited but also nervous about the challenges ahead.

        The first year was difficult. I had to learn so many new systems, and sometimes felt overwhelmed by the complexity. My manager was supportive though, and that made a big difference.

        By the second year, I was much more confident. I started taking on more responsibilities and even led a small project. That was really satisfying professionally.

        However, the third year brought some challenges. There was a reorganization, and I found myself in a new team with different dynamics. It took time to adjust, and honestly, I considered looking for opportunities elsewhere.

        What kept me here was the company culture and the meaningful work. I realized that despite the challenges, I was growing professionally and making valuable contributions.

        Now, looking at where I am, I feel proud of how far I've come. I've developed skills I never thought I would have, built strong relationships with colleagues, and contributed to important projects.

        For the future, I'm excited about the new initiatives we're starting. I hope to continue growing here and perhaps move into a leadership role eventually. The company has been supportive of professional development, and I appreciate that.

        When it comes to work-life balance, I think we could still improve. Sometimes the deadlines create stress that affects personal time, but I've gotten better at setting boundaries.

        Overall, I'm grateful for the opportunities I've had here and optimistic about what's ahead.
        """

        if use_sample:
            text_input = st.text_area("Interview Text", sample_text, height=300)
        else:
            text_input = st.text_area("Enter Interview Text", "", height=300)

        analyze_button = st.button("Analyze Text", type="primary")

        if analyze_button and text_input:
            with st.spinner("Analyzing text..."):
                # 1. Segment the text
                segments = segment_text(text_input, segmentation_method, segment_size)

                # Show segments
                st.subheader(f"Text Segmentation ({len(segments)} segments)")
                for i, segment in enumerate(segments):
                    with st.expander(f"Segment {i + 1}"):
                        st.write(segment)

                # 2. Analyze segments using the adaptor
                df, similarity_matrix = adaptor.analyze_segments(segments)

                # Store results in session state
                st.session_state.segments = segments
                st.session_state.df = df
                st.session_state.similarity_matrix = similarity_matrix

                st.success(f"Analysis complete! {len(segments)} segments analyzed.")

    with tab2:
        st.header("Analysis Results")

        if 'df' not in st.session_state:
            st.info("Please analyze text in the Text Input tab first.")
        else:
            df = st.session_state.df

            # Display data statistics
            st.subheader("Analysis Stats")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Total Segments", len(df))

            with col2:
                # Try to get sentiment counts
                sentiment_column = 'sentiment'
                positive_count = sum(df[sentiment_column] == "positive")
                neutral_count = sum(df[sentiment_column] == "neutral")
                negative_count = sum(df[sentiment_column] == "negative")

                dominant_sentiment = "positive" if positive_count > max(neutral_count, negative_count) else \
                    "neutral" if neutral_count > max(positive_count, negative_count) else \
                        "negative"

                st.metric("Dominant Sentiment", dominant_sentiment)

            # Display full DataFrame
            with st.expander("View Full Analysis Data"):
                st.dataframe(df)

            # Summary of sentiment
            st.subheader("Sentiment Summary")

            roberta_counts = df[sentiment_column].value_counts().reset_index()
            roberta_counts.columns = ["Sentiment", "Count"]

            col1, col2 = st.columns([1, 2])

            with col1:
                # Display sentiment counts
                st.write("Sentiment Counts:")
                st.dataframe(roberta_counts)

                # VADER stats
                st.write("VADER Stats:")
                vader_stats = {
                    "Stat": ["Mean", "Min", "Max"],
                    "Compound": [
                        df["vader_compound"].mean().round(2),
                        df["vader_compound"].min().round(2),
                        df["vader_compound"].max().round(2)
                    ]
                }
                st.dataframe(pd.DataFrame(vader_stats))

            with col2:
                # Simple bar chart of sentiment distribution
                import plotly.express as px
                fig = px.bar(
                    roberta_counts,
                    x="Sentiment",
                    y="Count",
                    color="Sentiment",
                    color_discrete_map={
                        "positive": "#2ca02c",
                        "neutral": "#d9d9d9",
                        "negative": "#d62728"
                    },
                    title="Sentiment Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

            # Download link
            st.subheader("Export Results")
            st.markdown(get_download_link(df), unsafe_allow_html=True)

    with tab3:
        st.header("Interactive Visualizations")

        if 'df' not in st.session_state or 'similarity_matrix' not in st.session_state:
            st.info("Please analyze text in the Text Input tab first.")
        else:
            df = st.session_state.df
            similarity_matrix = st.session_state.similarity_matrix
            segments = st.session_state.segments

            # 1. Similarity Matrix Heatmap
            st.subheader("Segment Similarity Matrix")
            st.markdown("""
            This heatmap shows the similarity between each pair of segments. 
            - Darker red indicates higher similarity
            - The diagonal is self-similarity (always 1.0)
            - Hover over cells to see more details
            """)

            try:
                sim_fig = plot_similarity_matrix(similarity_matrix, segments)
                st.plotly_chart(sim_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not plot similarity matrix: {e}")

                # Fallback simple heatmap
                import plotly.express as px
                fig = px.imshow(
                    similarity_matrix,
                    labels=dict(x="Segment", y="Segment", color="Similarity"),
                    x=[f"Seg {i + 1}" for i in range(len(segments))],
                    y=[f"Seg {i + 1}" for i in range(len(segments))],
                    color_continuous_scale="RdBu_r",
                    zmin=0, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)

            # 2. Sentiment Timeline
            st.subheader("Sentiment Analysis Timeline")
            st.markdown("""
            This chart shows sentiment changes across segments:
            - Blue line: VADER compound score (-1 to +1)
            - Square markers: Sentiment classification
            - Green background: positive sentiment region
            - Yellow background: neutral sentiment region
            - Red background: negative sentiment region
            """)

            try:
                # Adapt column names to match what plot_sentiment_timeline expects
                plot_df = df.copy()
                plot_df.rename(columns={
                    'sentiment': 'roberta_sentiment',
                    'sentiment_confidence': 'roberta_confidence'
                }, inplace=True)

                sent_fig = plot_sentiment_timeline(plot_df)
                st.plotly_chart(sent_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not plot sentiment timeline: {e}")

                # Simple fallback sentiment plot
                import plotly.graph_objects as go

                fig = go.Figure()

                # Add VADER compound score
                fig.add_trace(go.Scatter(
                    x=df["segment_id"],
                    y=df["vader_compound"],
                    mode="lines+markers",
                    name="VADER Compound",
                    line=dict(color="#1f77b4", width=2)
                ))

                # Add colored background for sentiment regions
                fig.add_shape(
                    type="rect", x0=-0.5, x1=len(df) - 0.5, y0=0.05, y1=1,
                    fillcolor="rgba(0,255,0,0.1)", line_width=0, layer="below"
                )
                fig.add_shape(
                    type="rect", x0=-0.5, x1=len(df) - 0.5, y0=-0.05, y1=0.05,
                    fillcolor="rgba(255,255,0,0.1)", line_width=0, layer="below"
                )
                fig.add_shape(
                    type="rect", x0=-0.5, x1=len(df) - 0.5, y0=-1, y1=-0.05,
                    fillcolor="rgba(255,0,0,0.1)", line_width=0, layer="below"
                )

                fig.update_layout(title="Sentiment Timeline", xaxis_title="Segment", yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)

            # 4. Similarity Scores
            st.subheader("Segment Similarity Analysis")
            st.markdown("""
            This chart shows how each segment relates to all other segments:
            - Orange line: Raw similarity score (average similarity to all other segments)
            - Green dashed line: Z-score of similarity
            - Red dotted lines: Statistical significance threshold (95% confidence)
            - Z-scores beyond ±1.96 indicate segments that are significantly different from others
            """)

            try:
                sim_score_fig = plot_similarity_scores(df)
                st.plotly_chart(sim_score_fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not plot similarity scores: {e}")

                # Simple fallback similarity plot
                import plotly.graph_objects as go

                fig = go.Figure()

                # Add raw similarity
                fig.add_trace(go.Scatter(
                    x=df["segment_id"],
                    y=df["similarity_raw"],
                    mode="lines+markers",
                    name="Raw Similarity",
                    line=dict(color="#ff7f0e")
                ))

                # Add z-scores if available
                if "similarity_z_score" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df["segment_id"],
                        y=df["similarity_z_score"],
                        mode="lines+markers",
                        name="Z-Score",
                        line=dict(color="#2ca02c", dash="dash")
                    ))

                fig.update_layout(title="Segment Similarity", xaxis_title="Segment", yaxis_title="Score")
                st.plotly_chart(fig, use_container_width=True)




if __name__ == "__main__":
    main()
