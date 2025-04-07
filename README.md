# 🏦 Loan Approval Prediction System

This is a Machine Learning-based project that predicts whether a loan will be approved or not based on various applicant features. The model is trained on a dataset of loan applications and uses different classification algorithms to determine the best-performing model.

---

## 📌 Features

- Data Preprocessing
- Exploratory Data Analysis (EDA)
- Model Training with:
  - Decision Tree Classifier
  - Logistic Regression
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
- Model Evaluation using Accuracy Score and Cross-Validation
- Scikit-learn Pipelines
- Visualizations using Matplotlib and Seaborn

---

## 🛠️ Requirements

Install the following Python libraries before running the project:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

---

## 📂 Dataset

The dataset used for training and testing should include the following columns (example):

- Gender
- Married
- Dependents
- Education
- Self_Employed
- ApplicantIncome
- CoapplicantIncome
- LoanAmount
- Loan_Amount_Term
- Credit_History
- Property_Area
- Loan_Status (Target Variable)

---

## 🧠 Libraries Used

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as py
import seaborn as sns
import warnings
```

---

## 🧪 Model Evaluation

Each model is evaluated using:

- Accuracy Score
- Visualization of model performance

---

## 📊 Exploratory Data Analysis (EDA)

The project includes a comprehensive EDA to uncover patterns and correlations using:

- Box Plots
- Count Plots for categorical data

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/loan-approval-prediction.git
   cd loan-approval-prediction
   ```

2. Run the Jupyter Notebook or Python script:
   ```bash
   jupyter notebook Loan_Approval_Prediction.ipynb
   ```


---

## 📈 Result

The system displays the performance of all trained models and highlights the best one for loan approval prediction based on accuracy and cross-validation score.

---

## 📃 License

This project is licensed under the MIT License.

---

## 🙋‍♂️ Author

**Your Name**  
[GitHub Profile](https://github.com/sayed2174)

