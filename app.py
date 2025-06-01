import streamlit as st
from utils import motorcycle_utils, sentiment_analysis_utils, wikipedia_topic_clusterer_utils

st.set_page_config(page_title="ML Showcase", layout="wide")
st.markdown("<h2>Machine Learning Streamlit Showcase</h2>", unsafe_allow_html=True)


tab1, tab2, tab3 = st.tabs(["🧮 Accident Probability (Regression)",
                            "💬 Sentiment Analysis (Classification)",
                            "🧠 Wikipedia Topic Clusterer (Clustering)"])

with tab1:
    motorcycle_utils.render_regression_tab()

with tab2:
    sentiment_analysis_utils.render_sentiment_analysis_tab()

with tab3:
    wikipedia_topic_clusterer_utils.render_wikipedia_topic_clusterer_tab()
