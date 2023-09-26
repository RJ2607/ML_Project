import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to a pandas DataFrame
sonar_data = pd.read_csv('sonar rock vs mine\sonar data.csv', header=None)
print(sonar_data.head())
