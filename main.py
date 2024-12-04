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
from xgboost import XGBClassifier
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


#Checking the class distribution of target column
df.describe()

def plot_histogram(df, column_name):
    plt.figure(figsize=(5,4))
    sns.histplot(df[column_name] , kde=True)
    plt.title(f"Distribution of {column_name}")
    
    col_mean = df[column_name].mean()
    col_median = df[column_name].median()
    
    plt.axvline(col_mean , color="red" , linestyle="--" , label="Mean")
    plt.axvline(col_median , color="blue" , linestyle="-" , label="Median")
    
    plt.legend()
    plt.show()
    
plot_histogram(df, "tenure")
plot_histogram(df, "TotalCharges")


#Box plot for numerical features
def plot_boxplot(df, column_name):
    plt.figure(figsize=(5,3))
    sns.boxplot(y=df[column_name])
    plt.title(f"Box Plot of {column_name}")
    plt.ylabel(column_name)
    plt.show()
    
plot_boxplot(df, "tenure") 
plot_boxplot(df, "MonthlyCharges")   
plot_boxplot(df, "TotalCharges")


#Correlation matrix - heatmap
plt.figure(figsize=(8,4))
sns.heatmap(df[["tenure", "MonthlyCharges", "TotalCharges"]].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#Countplot for categorical columns and converting a column
object_cols = df.select_dtypes(include="object").columns.tolist()
object_cols = ["SeniorCitizen"] + object_cols

for col in object_cols:
    plt.figure(figsize=(5,3))
    sns.countplot(x=df[col])
    plt.title(f"Count Plot of {col}")
    plt.show()
    
df["Churn"] = df["Churn"].replace({"Yes": 1 , "No":0})


#Label encoding of categorical fetaures
object_columns = df.select_dtypes(include="object").columns.tolist()

encoders = {}
for column in object_columns:
    label_encoder = LabelEncoder()
    df[column] = label_encoder.fit_transform(df[column])
    encoders[column] = label_encoder

with open("encoders.pkl" , "wb") as f:
    pickle.dump(encoders, f)


#Splitting data and training sets
X = df.drop(columns="Churn")
y = df["Churn"]

X_train, X_test, y_train, y_test  = train_test_split(X, y , test_size=0.2 , random_state=42)


#Synthetic Minority Oversampling TEchnique (SMOTE)
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


#Training with default hyperparameters
models = {
    "Desicion Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42)
}

cv_scores = {}

for model_name, model in models.items():
    print(f"Training {model_name} with default parameters")
    scores = cross_val_score(model, X_train_smote, y_train_smote, cv = 10 , scoring="accuracy")
    cv_scores[model_name] = scores
    print(f"{model_name} cross-validation accuracy: {np.mean(scores):.2f}")
    print("-"*70)
    

#Using Random Forest
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote , y_train_smote)


# Model Evaluation
y_test_pred = rfc.predict(X_test)

print("Accuracy Score:\n", accuracy_score(y_test, y_test_pred))
print("Confsuion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))


#Save the trained model as a pickle file
model_data= {"model": rfc , "features_names":X.columns.tolist()}

with open("customer_churn_model.pkl" , "wb") as f:
    pickle.dump(model_data , f)
    

#Load teh saved model and the feature names 
with open("customer_churn_model.pkl" , "rb") as f:
    model_data = pickle.load(f)

loaded_model = model_data["model"]
features_nams = model_data["features_names"]   


#Trying an extra input
input_data = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

input_data_df = pd.DataFrame([input_data])

with open("encoders.pkl" , "rb") as f:
    encoders = pickle.load(f)

for column, encoder in encoders.items():
    input_data_df[column] = encoder.transform(input_data_df[column])

prediction = loaded_model.predict(input_data_df)
pred_prob = loaded_model.predict_proba(input_data_df)

print(prediction)

print(f"Prediction: {"Churn" if prediction[0] == 1 else "No Churn"}")
print(f"Prediction Probability: {pred_prob}" )