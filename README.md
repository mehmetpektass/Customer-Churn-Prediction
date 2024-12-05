# Customer Churn Prediction 🤔

## Description
This project aims to predict customer churn for a telecom company using machine learning models. The dataset used is the WA_Fn-UseC_-Telco-Customer-Churn.csv file, which contains customer information and service usage details. The pipeline involves data preprocessing, feature engineering, model training, and evaluation.

<br>

## Features

* **Data Preprocessing:** Handling missing and erroneous data,
encoding categorical variables using LabelEncoder,
removing unnecessary features (e.g., customerID),
addressing class imbalance using SMOTE.
* **Feature Engineering:** Histogram and boxplot visualizations for numerical columns, correlation heatmaps for feature selection.
*  **Machine Learning Models:** Models trained: Decision Tree, Random Forest, and XGBoost, hyperparameter tuning using cross-validation.
* **Model Evaluation:** Metrics: Accuracy, Confusion Matrix, Classification Report, predicted churn probability for new inputs.
* **Pickle Integration:** Encoders and the trained model are saved and loaded using pickle for future predictions.




<br>

## Tech Stack
### Tools and Libraries
* **Python:** Core programming language.
* **Pandas:** Data manipulation and preprocessing.
* **NumPy:** Numerical computations.
* **Matplotlib & Seaborn:** Data visualization.
* **Scikit-learn:** Machine learning library for training and evaluation.
* **Imbalanced-learn:** For oversampling techniques like SMOTE.
* **XGBoost:** Gradient-boosted decision trees for classification.
* **Pickle:** Model serialization and persistence.

### Key Algorithms
* **Decision Tree Classifier:** Simple, interpretable tree-based model.
* **Random Forest Classifier:** Ensemble learning technique for improved accuracy.
* **XGBoost Classifier:** High-performance gradient boosting.



<br>

<br>


## Installation & Setup
```bash
git clone https://github.com/MehmetPektas/Telco-Customer-Churn.git
cd Telco-Customer-Churn

```
```
pip install -r requirements.txt

```
### Add Dataset

Place the WA_Fn-UseC_-Telco-Customer-Churn.csv file in the project root directory.

<br>


## Contribution Guidelines  🚀
 Pull requests are welcome. If you'd like to contribute, please:

* Fork the repository.
* Create a feature branch.
* Submit a pull request with a clear description of changes.



