import streamlit as st
from utils import motorcycle_utils

st.set_page_config(page_title="ML Showcase", layout="wide")
st.title("🚀 Machine Learning Streamlit Showcase")

tab1, tab2, tab3 = st.tabs(["🧮 Accident Probability (Regression)",
                            "💬 Sentiment Analysis",
                            "🧠 Word Clustering"])

with tab1:
    motorcycle_utils.render_regression_tab()

# with tab2:
#     sentiment_utils.render_sentiment_tab()

# with tab3:
#     clustering_utils.render_clustering_tab()
