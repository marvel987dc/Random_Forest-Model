import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn
import seaborn

#reading the CSV file
df_juan = pd.read_csv(r'C:\Users\barre\Documents\Semester 4 (Current)\Supervised Learning (SEC. 001)\Week_14\JuanBarrero_COMP247_assignment5\Data\pima-indians-diabetes (1).csv')

#naming the columns
df_juan.columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

#printuing column names and types
print("Columns names and Data Types: ")
print(df_juan.dtypes)

#printing the Nan values (missing values)
print("\nMissing Values: ")
columns_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_juan[columns_with_invalid_zeros] = df_juan[columns_with_invalid_zeros].replace(0, pd.NA)
print(df_juan[columns_with_invalid_zeros].eq(0).sum())

#print the description statistics (min, max, avg, std, count, etc)
print("\nDescription statistics: ")
print(df_juan.describe())

#showing the categorical values
print("Categorical values in 'outcome': ")
print(df_juan['Outcome'].value_counts())

#pre-processing and training
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#spliting the data (separate features from the target class)

columns_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_juan[columns_with_invalid_zeros] = df_juan[columns_with_invalid_zeros].replace(0, pd.NA)

X = df_juan.drop('Outcome', axis=1)
y = df_juan['Outcome']

seed = 42
X_train_Juan, X_test_Juan, y_train_Juan, y_test_Juan = train_test_split(X, y, test_size=0.3, random_state=seed)

transformer_Juan = StandardScaler()

X_train_Juan = transformer_Juan.fit_transform(X_train_Juan)
X_test_Juan = transformer_Juan.fit_transform(X_test_Juan)

print("X_train_juan shape:", X_train_Juan.shape)
print("X_test_juan shape:", X_test_Juan.shape)
