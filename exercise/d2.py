import numpy as np
import pandas as pd
df = pd.read_csv('wineq.csv', sep = ';')
x = df.drop(columns = 'quality').to_numpy()
y = df.iloc[:, -1].to_numpy()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
print(x_train)
print(y_train)
print(x_test)
print(y_test)