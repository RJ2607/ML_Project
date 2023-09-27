import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn import svm
from sklearn.metrics import accuracy_score

#data loading
diabetes_dataset = pd.read_csv('Diabetes prediction\diabetes.csv')

#print(diabetes_dataset['Outcome'].value_counts())
#0    500
#1    268

X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data