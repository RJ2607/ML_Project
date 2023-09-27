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

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_train_accuracy = classifier.predict(X_train)
accuracy_test = accuracy_score(X_train_accuracy, Y_train)

X_test_accuracy = classifier.predict(X_test)
accuracy_test = accuracy_score(X_test_accuracy, Y_test)

#print("accuracy on test data: ",accuracy_test)

#predictive system
#input data
input_data = (7,187,68,39,304,37.7,0.254,41)
input_data_as_numpy_array = np.asarray(input_data)
 #reshape numpy array
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
scaler.fit(input_data_reshape)
input_data_reshape = scaler.transform(input_data_reshape)

#prediction
prediction = classifier.predict(input_data_reshape)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")