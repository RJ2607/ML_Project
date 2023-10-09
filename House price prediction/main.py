import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

house_price_dataset = sklearn.datasets.load_boston()
#print(house_price_dataset)

#structuring data from pandas

house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns = house_price_dataset.feature_names)
#print(house_price_dataframe.head())

#adding target value in dataframe
house_price_dataframe['price'] = house_price_dataset.target
#print(house_price_dataframe.head())
#print(house_price_dataframe.isnull().sum()) (checking for missing values)

#establishing relation between features
correlation = house_price_dataframe.corr()

#plotting heatmap
#plt.figure(figsize=(10,10))
#sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot = True, annot_kws={'size':8}, cmap='Blues')
#plt.show()

#spliting data and target
X = house_price_dataframe.drop('price', axis = 1)
Y = house_price_dataframe['price']

#splitting the data into training data and test data
X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

#Model training

model = XGBRegressor()

model.fit(X_train,Y_train)

#prediction
training_data_prediction = model.predict(X_train)

#print(training_data_prediction)
#R squared error

score_1 = metrics.r2_score(Y_train,training_data_prediction)

#mean absolute error

score_2 = metrics.mean_absolute_error(Y_train,training_data_prediction)

#print(score_1)
#print(score_2)

#test data error
test_data_prediction = model.predict(X_test)
score_3 = metrics.r2_score(Y_test,test_data_prediction)
score_4 = metrics.mean_absolute_error(Y_test,test_data_prediction)

#print(score_3)
#print(score_4)

#visualizing actual price on predicted price
plt.scatter(Y_train, training_data_prediction)
plt.xlabel('actual')
plt.ylabel('predicted')
plt.title('title')
plt.show()