import streamlit as st
import pandas as pd
import joblib
import plotly.express as px


def render_sentiment_analysis_tab():
    @st.cache_resource
    def load_model():
        with open("models/sentiment_predictor.pkl", "rb") as f:
            model = joblib.load(f)
        return model

    loaded_model = load_model()

    st.markdown("""
    <div style='padding:15px; border-radius:10px; margin-bottom:20px; background-color:#f9f9f9;'>
        <h2>Sentiment Analysis App</h2>
        <p style='font-size:16px;'>
            This application predicts the sentiment of product reviews using a Logistic Regression model trained on 
            <a href='https://www.kaggle.com/datasets/niraliivaghani/flipkart-product-customer-reviews-dataset' target='_blank'>200,000+ Flipkart product reviews</a>.
            It analyzes textual input and classifies it as <strong>positive</strong>, <strong>neutral</strong>, or <strong>negative</strong> sentiment.
        </p>
        <p style='font-size:16px;'>
            <strong>Logistic Regression</strong> was selected after comparing multiple classification models including Naive Bayes, SVM, and XGBoost. It achieved an impressive 
            <strong>92% accuracy</strong> on the test dataset.
        </p>
        <p style='font-size:16px;'>
            Enter a product review below to instantly see the predicted sentiment along with a visual confidence score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    
    review = st.text_area("Enter a product review:", "Samsung Galaxy S23 Ultra is a great phone with amazing camera quality and performance.", height=150)

    if st.button("Predict Sentiment"):
        # Predict sentiment
        prediction = loaded_model.predict([review])
        probabilities = loaded_model.predict_proba([review])[0]
        classes = loaded_model.classes_

        predicted_sentiment = prediction[0].capitalize()

        sentiment_color_map = {
            "Positive": "green",
            "Neutral": "gray",
            "Negative": "red"
        }

        color = sentiment_color_map.get(predicted_sentiment, "black")

        st.markdown(f"""
        <div style='background-color:#f0f8ff; padding:10px; border-radius:10px;'>
            <h2 style='color:#007acc;'>Predicted Sentiment:</h2>
            <h1 style='color:{color}; font-weight:bold'>{predicted_sentiment}</h1>
            <p style='font-style: italic;'>Thank you for your feedback!</p>
        </div>
        """, unsafe_allow_html=True)

        # Create a Plotly horizontal bar chart
        prob_data = {
            "Sentiment": [c.capitalize() for c in classes],
            "Probability": [round(p, 4) for p in probabilities]
        }

        fig = px.bar(
            prob_data,
            x="Probability",
            y="Sentiment",
            orientation="h",
            text="Probability",
            color="Sentiment",
            color_discrete_map={
                "Negative": "red",
                "Neutral": "gray",
                "Positive": "green"
            }
        )

        fig.update_traces(
            texttemplate='%{text:.2%}',
            textposition='outside',
            insidetextanchor='start',
            width=0.3,
            textfont=dict(size=14, color='black')
        )

        fig.update_layout(
            width=600,
            height=300,
            title="Sentiment Probabilities",
            xaxis=dict(
                range=[0, 1],
                showticklabels=False,
                title=None
            ),
            yaxis=dict(
                autorange="reversed",
                title=None,
                tickfont=dict(size=14, color='black')
            ),
            showlegend=False,
            plot_bgcolor='white',
            bargroupgap=0.05,
            # margin=dict(t=40, b=20, l=40, r=100)  # Optimized spacing
        )

        st.plotly_chart(fig, use_container_width=False)





