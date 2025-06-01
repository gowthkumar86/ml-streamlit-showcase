## Project Overview

This repository contains three machine learning-powered web applications designed to provide actionable insights through prediction, analysis, and content clustering:

* **Motorcycle Risk Predictor:** Estimates the accident risk for motorcycle riders based on safety and environmental factors.
* **Sentiment Analysis Predictor:** Analyzes sentiment from user-submitted text to help businesses understand customer opinions.
* **Wikipedia Topic Clusterer:** Clusters Wikipedia article sentences into meaningful topics using hierarchical clustering and text embeddings, helping users explore complex content in an organized way.

All apps feature intuitive Streamlit interfaces, fast model inference, and clear visual feedback to improve decision-making and user experience.

---

## Why I Created This Project

After completing a comprehensive machine learning course on Udemy ([link here](https://www.udemy.com/course/machinelearning)), I wanted to put the concepts and techniques I learned into practice. Building this project gave me the opportunity to apply theory to real-world scenarios, deepen my understanding, and gain hands-on experience. It also allowed me to create practical tools that solve meaningful problems, reinforcing my skills while delivering value.

---

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

# Sentiment Analysis

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

# Wikipedia Topic Clusterer

**Interactive tool that extracts, embeds, and clusters text from Wikipedia articles using Sentence Transformers and Hierarchical Clustering.**
Ideal for analyzing large text sections and organizing them into meaningful topic groups.

## Features

* Scrapes Wikipedia content from any topic URL
* Cleans and preprocesses the text
* Splits content into meaningful sentences
* Generates sentence embeddings with `sentence-transformers`
* Clusters similar sentences into up to **50 topic groups** using hierarchical clustering
* Displays:

  * Topic clusters (expandable UI)
  * Full DataFrame of clusters
  * Number of clusters detected
* Built using `Streamlit` with real-time interactive updates

## Input

* A valid Wikipedia article URL

## How It Works

1. The user enters a Wikipedia URL
2. The app fetches and cleans the article content
3. Sentences are embedded using a transformer model (e.g., `all-MiniLM-L6-v2`)
4. Cosine distances are used to perform hierarchical clustering
5. A maximum of **50 clusters** is formed using silhouette analysis and cut-off heuristics
6. The result:

   * Expandable UI for topic clusters
   * Pandas DataFrame showing all clusters
   * Total number of clusters

## Why Use This App?

* Turn long articles into digestible topic clusters
* Explore semantic structure of large texts easily
* Ideal for educational use, summarization, and knowledge extraction
