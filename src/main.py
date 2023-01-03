import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


with open('data/diabetes.csv', 'rb') as f:
    data_base = pd.read_csv(f)

x_diabetes = data_base.iloc[:, 1:7].values
y_diabetes = data_base.iloc[:, 8].values

scaller_diabetes = StandardScaler()
x_diabetes = scaller_diabetes.fit_transform(x_diabetes)

x_diabetes_training, x_diabetes_test, y_diabetes_training, y_diabetes_test = train_test_split(x_diabetes, y_diabetes, test_size=0.2, random_state=0)