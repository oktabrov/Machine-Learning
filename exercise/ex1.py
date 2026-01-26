import numpy as np
import pandas as pd

dt = pd.read_csv('iris.csv')
x = dt.drop(columns='target').values
y = dt.iloc[:, -1].to_numpy()
print(x)
print(y)