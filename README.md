# House Price Prediction Using Machine Learning

## 📖 Project Description
This project focuses on predicting house prices in California using machine learning. It leverages the **California Housing Dataset** from `scikit-learn` and applies the powerful **XGBoost Regressor** algorithm for prediction. The project includes data exploration, correlation analysis, model training, and performance evaluation.

---

## 📊 Dataset
- **California Housing Dataset**:
    - 8 features: Median Income, House Age, Average Rooms, Average Bedrooms, Population, Average Occupancy, Latitude, Longitude.
    - Target: Median House Value (in $100,000).
- ⚠️ The Boston Housing Dataset has been deprecated due to ethical concerns and is not used in this project.

---

## 🔍 Exploratory Data Analysis (EDA)
- Visualizes feature correlations with a heatmap.
- Helps understand which features are most relevant to house prices.

---

## 🚀 Model Used
- **XGBoost Regressor**:
    - Gradient boosting method optimized for performance.
    - Trained on the dataset after a train-test split.

---

## ✅ Evaluation Metrics
- Mean Squared Error (MSE) is used to evaluate model performance.
- (Optional) R² score can be calculated for accuracy.

---

## ⚙️ Dependencies
Make sure to install the required libraries before running the notebook:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## 🎯 How to Run
1. Clone the repository:
```bash
git clone https://github.com/krishnakantt/House-Price-Prediction.git
cd house-price-prediction
```
2. Open the Jupyter Notebook:
```bash
jupyter notebook House\ Price\ Prediction.ipynb
```
3. Run the cells in order:
- Load dataset
- Perform EDA (including correlation heatmap)
- Train XGBoost model
- Evaluate model performance

---

## 📈 Example Visualization
```bash
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)

correlation = X.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
plt.show()
```

---

## 📚 References
- California Housing Dataset Documentation
- XGBoost Documentation
