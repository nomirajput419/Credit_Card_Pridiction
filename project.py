import numpy as np
import pandas as pd 

data = pd.read_csv('credit_card.csv')
# print(data)

# idr hum ne ek row khatam kiya ha jo hamare credit card data ka first pa tha x, x1 etc
data.columns = data.iloc[0]
data = data.drop(0).reset_index(drop=True)


# mein ne target column ka name change kiya ha 
data = data.rename(columns={"default payment next month": "target"})

# <> Data Processing:
# Handel Missing value:

# missing values check ki ha 
# print(data.isnull().sum())

# Encode categorical variables:

# Unique values check ki hain categorical variables ka leye 
# print("SEX:", data["SEX"].unique())
# print("EDUCATION:", data["EDUCATION"].unique())
# print("MARRIAGE:", data["MARRIAGE"].unique())

# Split the dataset into training and testing sets:
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x = data.drop('target', axis=1)
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features:
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train) 
x_test_scaled = scaler.transform(x_test) 

# <> Model Development:
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, random_state=42)

log_reg.fit(x_train_scaled, y_train)

y_pred = log_reg.predict(x_test_scaled)           
y_pred_prob = log_reg.predict_proba(x_test_scaled)[:, 1]  


# Model Evaluation:
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

acc = accuracy_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_pred_prob)

report = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("ROC-AUC:", roc_auc)
print("\nClassification Report:\n", report)
print("\nConfusion Matrix:\n", cm)

