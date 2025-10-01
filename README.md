 Credit Card Default Prediction

This project focuses on predicting credit card default risk using Logistic Regression. It preprocesses financial and demographic data, scales numerical features, and evaluates the model with multiple metrics.

 Features

Data Preprocessing (handling missing values, renaming columns, encoding)

Data Scaling using StandardScaler

Model Training with Logistic Regression

Model Evaluation with:

Accuracy

ROC-AUC Score

Confusion Matrix

Classification Report

 Dataset

The dataset contains customer demographic and financial details, along with a target variable indicating default payment next month.

 Installation

Clone the repository:

git clone https://github.com/nomirajput419/credit-card-default-prediction.git
cd credit-card-default-prediction


Install dependencies:

pip install -r requirements.txt


(requirements.txt: pandas, numpy, scikit-learn)

 Usage

Run the script:

python credit_card_default.py


This will:

Load and preprocess the dataset

Train the Logistic Regression model

Print evaluation metrics (Accuracy, ROC-AUC, Confusion Matrix, Report)

 Example Output
Accuracy: 0.82
ROC-AUC: 0.77
Classification Report:

Confusion Matrix:


Tech Stack

Python

Pandas, NumPy

Scikit-learn
