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

X = diabetes_dataset.drop(columns = 'Outcome', axis=1)
Y = diabetes_dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, stratify=Y, random_state=2)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

#print("accuracy on test data: ",accuracy_test)

#predictive system
#input data
#input_data = [8,176,90,34,300,33.7,0.467,58]
input_data = []

i=0
while(i<8):
    if (i==0):
        a = float(input("Enter no. of Pregnancies: "))
    elif(i==1):
        a = float(input("Enter Glucose level: "))
    elif(i==2):
        a = float(input("Enter Blood Pressure: "))
    elif(i==3):
        a = float(input("Enter SkinThickness: "))
    elif(i==4):
        a = float(input("Enter Insulin: "))
    elif(i==5):
        a = float(input("Enter BMI: "))
    elif(i==6):
        a = float(input("Enter DiabetesPedigreeFunction: "))
    elif(i==7):
        a = float(input("Enter Age: "))

    input_data.append(a)
    a = 0
    i+=1

print(input_data)
input_data_as_numpy_array = np.asarray(input_data)
 #reshape numpy array
input_data_reshape = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data

std_data = scaler.transform(input_data_reshape)

#prediction
prediction = classifier.predict(std_data)

if (prediction[0] == 0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")