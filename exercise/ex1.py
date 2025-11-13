import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

dataset = pd.read_csv('Data.csv')
x1 = dataset.drop(columns = 'Purchased').to_numpy()
y1 = dataset.iloc[:, -1].to_numpy()
print(x1)
print()
impute = SimpleImputer(missing_values = np.nan, strategy = 'mean', fill_value = 1)
x1[:, 1:3] = impute.fit_transform(x1[:, 1:3])
print(x1)
print()
c_t = ColumnTransformer(transformers = [
    ('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
x1 = c_t.fit_transform(x1)
print(x1)
print()
le = LabelEncoder()
y1 = le.fit_transform(y1)
print(y1)
print()
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = .2, random_state = 1)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
print()
