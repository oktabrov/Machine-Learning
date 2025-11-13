import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
df = pd.read_csv('iris.csv').to_numpy()
print(df)
print()
##c_t = ColumnTransformer(transformers = [('l', OneHotEncoder(), slice(0, -1))], remainder = 'passthrough')
##df = c_t.fit_transform(df)
##print(df)
##print()
##df = df.toarray()
x_train, x_test, y_train, y_test = train_test_split(df[:, :-1], df[:, -1], test_size = .2, random_state = 1)
print(x_test)
print(y_test)
print()
##df = df.toarray()  # before train_test_split
se = StandardScaler()
x_train = se.fit_transform(x_train)
x_test = se.transform(x_test)