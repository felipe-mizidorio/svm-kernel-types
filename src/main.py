import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

with open('data/diabetes.csv', 'rb') as f:
    data_base = pd.read_csv(f)

x_diabetes = data_base.iloc[:, 1:7].values
y_diabetes = data_base.iloc[:, 8].values

scaller_diabetes = StandardScaler()
x_diabetes = scaller_diabetes.fit_transform(x_diabetes)

x_diabetes_training, x_diabetes_test, y_diabetes_training, y_diabetes_test = train_test_split(x_diabetes, y_diabetes, test_size=0.2, random_state=0)

svm_diabetes = SVC(C=2.0, kernel='linear', random_state=1) # kernel = 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
svm_diabetes.fit(x_diabetes_training, y_diabetes_training)
predictions = svm_diabetes.predict(x_diabetes_test)

print(accuracy_score(y_diabetes_test, predictions))