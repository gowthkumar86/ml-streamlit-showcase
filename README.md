# Motorcycle Risk Predictor

**A Streamlit web application to estimate the accident risk for motorcycle riders using machine learning.**
This app considers various safety and environmental factors to provide a risk score that helps users make safer riding decisions.

---

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

---

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

---

## How It Works

1. The user inputs relevant information about their riding scenario.
2. The data is fed into a pre-trained machine learning model (`motorcycle-risk-predictor.pkl`).
3. The app outputs a **risk score in percentage** with color indicators and a safety reminder.

---

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gowthkumar86/ml-streamlit-showcase.git
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```


---

## 🧪 Model Training (Optional)

If you're interested in training the model yourself, check out the `utils/` folder (if included) for Jupyter notebooks and training scripts.

---

## 🙌 Acknowledgements

* Built using [Streamlit](https://streamlit.io)
* Model trained using scikit-learn and pandas
* Inspired by real-world road safety analysis
