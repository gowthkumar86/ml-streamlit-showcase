# Motorcycle Risk Predictor

**Web application to estimate the accident risk for motorcycle riders using machine learning.**
This app considers various safety and environmental factors to provide a risk score that helps users make safer riding decisions.



## Features

* Predicts accident probability based on rider inputs
* Built using a trained regression model
* Clean, interactive Streamlit interface
* Color-coded risk level:

  * ✅ **Low Risk** (Green)
  * ⚠️ **Moderate Risk** (Yellow)
  * 🔴 **High Risk** (Red)
* Responsive and user-friendly design
* Includes model caching for fast performance



## Input Parameters

Users enter the following details:

* Rider age
* Years of riding experience
* Speed (km/h)
* Helmet usage (Yes/No)
* Alcohol detected (Yes/No)
* Recent traffic violations
* Road surface condition
* Lighting condition
* Brake condition (scale 0 to 1)
* Weather condition
* Road type



## How It Works

1. The user inputs relevant information about their riding scenario.
2. The data is fed into a pre-trained machine learning model (`motorcycle-risk-predictor.pkl`).
3. The app outputs a **risk score in percentage** with color indicators and a safety reminder.

---

# Sentiment Analysis Predictor

**Web application to analyze sentiment from user reviews or text inputs using a trained classification model.**
This app helps users quickly understand the emotional tone behind texts, enabling smarter business and product decisions.



## Features

* Predicts sentiment category (Positive, Neutral, Negative) from text input

* Built using a robust trained classification model

* Clean, interactive Streamlit interface

* Color-coded sentiment levels:

  * ✅ **Positive** (Green)
  * ⚪ **Neutral** (Gray)
  * 🔴 **Negative** (Red)

* Displays confidence scores with intuitive horizontal bar charts

* Responsive, user-friendly design with fast predictions and caching support



## Input Parameters

Users provide:

* Product or service review text
* Customer feedback
* Any short to medium-length textual data for sentiment evaluation



## How It Works

1. The user enters or pastes their text input.
2. The text is processed and sent to the pre-trained sentiment analysis model (`sentiment_predictor.pkl`).
3. The app predicts the sentiment category and displays:

   * The predicted sentiment with color-coded styling
   * Confidence scores for each sentiment class via a horizontal bar chart



## Why Use This App?

* Gain instant insights into customer opinions and market sentiment
* Make data-driven decisions in marketing, product development, and customer service
* Automate sentiment extraction without manual analysis

---
