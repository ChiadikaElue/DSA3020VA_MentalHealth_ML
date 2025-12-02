# Mental Health Risk Classification from Social Media Text

### Group Members
Chiadika Elue 670280
Yar Deng Kuot 669215
Abdirisak Hussein A 668776

## Overview
This project uses machine learning to classify mental health–related text into Positive, Negative, or Neutral sentiments. It follows CRISP-DM methodology and includes preprocessing, modeling, evaluation, and deployment.

## Features
- Text cleaning + preprocessing
- TF-IDF vectorization
- ML models: Logistic Regression, SVM, Naive Bayes, Random Forest
- Evaluation + confusion matrices
- Flask API
- Streamlit App interface

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Run Jupyter Notebooks
Inside `notebooks/` run:
- 01_data_preprocessing.ipynb
- 02_model_development.ipynb
- 03_model_evaluation_and_saving.ipynb

### 3. Run Flask API
python app/app.py

### 4. Run Streamlit interface
streamlit run app/streamlit_app.py

## Project Structure
mental-health-ml/
│
├── data/
│   └── sentiment_analysis.csv
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_model_evaluation_and_saving.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── utils.py
│
├── app/
│   ├── app.py
│   └── streamlit_app.py
│
├── models/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── README.md
└── requirements.txt

