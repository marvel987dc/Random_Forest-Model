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
    'Insulin', 'BIM', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

#printuing column names and types
print("Columns names and Data Types: ")
print(df_juan.dtypes)

#printing the Nan values (missing values)
print("\nMissing Values: ")
columns_with_invalid_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BIM']
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

#spliting the data (separate features from the target class)
X = df_juan.drop('Outcome', axis=1)
y = df_juan['Outcome']

transformer_Juan = StandardScaler()

X_scaled = transformer_Juan.transform(X)
