import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBRFClassifier
import pickle

# Load the data to a pandas dataframe

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Check the data
df.shape
df.head()
df.info()

# Dropping customerID column as this is not required for modelling
df = df.drop(columns=["customerID"])

# printing the unique values in all the columns
numerical_features_list = ["tenure", "MonthlyCharges", "TotalCharges"]
for col in df.columns:
    if col not in numerical_features_list:
        print(col, df[col].unique())
        print("-"*50)
        
#Deleting non values of total charges
df[df["TotalCharges"] == " "]

#Converting object to float for total charges
df["TotalCharges"] = df["TotalCharges"].replace({" " : "0.0"})

df["TotalCharges"] = df["TotalCharges"].astype(float)

