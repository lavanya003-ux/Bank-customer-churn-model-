# Bank-customer-churn-model
#### Title of Project
*Bank Customer Churn Prediction*

#### Objective
To predict which customers are likely to churn (leave the bank) based on historical data.

#### Data Source
The dataset used in this project is assumed to be a CSV file named customer_data.csv.

#### Import Libraries
python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import xgboost as xgb


#### Import Data
python
data = pd.read_csv('customer_data.csv')


#### Describe Data
python
print(data.head())
print(data.info())
print(data.describe())
print(data.isnull().sum())


#### Data Visualization
python
# Churn Distribution
sns.countplot(x='Churn', data=data)
plt.title('Churn Distribution')
plt.show()

# Age Distribution
sns.histplot(data['Age'].dropna(), bins=30)
plt.title('Age Distribution')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()


#### Data Preprocessing
python
# Handle Missing Values
data.fillna(data.mean(), inplace=True)

# Encode Categorical Variables
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'])

# Feature Engineering
data['BalanceSalaryRatio'] = data['Balance'] / data['EstimatedSalary']


#### Define Target Variable (y) and Feature Variables (X)
python
X = data.drop(['Churn'], axis=1)
y = data['Churn']


#### Train Test Split
python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#### Modeling
python
# Standardize Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Model (example with XGBoost)
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)


#### Model Evaluation
python
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'ROC-AUC: {roc_auc}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
                           param_grid=param_grid,
                           cv=3,
                           scoring='roc_auc')
grid_search.fit(X_train, y_train)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best ROC-AUC: {grid_search.best_score_}')


#### Prediction
python
# Retrain with best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Final Evaluation
y_pred_best = best_model.predict(X_test)
roc_auc_best = roc_auc_score(y_test, y_pred_best)
print(f'Final ROC-AUC: {roc_auc_best}')


#### Explanation
The bank customer churn model project aims to identify customers at risk of leaving the bank. By leveraging historical data, we can make informed predictions and take proactive measures to improve customer retention. The steps include data exploration, preprocessing, model training, evaluation, and prediction. The model’s performance is evaluated using metrics such as accuracy and ROC-AUC. Hyperparameter tuning is conducted to optimize the model's performance. The final model provides actionable insights that can help the bank implement targeted retention strategies.
