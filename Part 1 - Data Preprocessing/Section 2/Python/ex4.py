import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
df = pd.read_csv('Data.csv')
x = df.iloc[:, :-1].to_numpy()
y = df.iloc[:, -1].to_numpy()
imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value = 12)
x[:, 1:] = imputer.fit_transform(x[:, 1:])
c_t = ColumnTransformer(transformers = [('label', OneHotEncoder(), [0])], remainder = 'passthrough')
x = c_t.fit_transform(x)
le = LabelEncoder()
y = le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print(x_test)
print(y_test)
print()
print(x_train)
print(y_train)