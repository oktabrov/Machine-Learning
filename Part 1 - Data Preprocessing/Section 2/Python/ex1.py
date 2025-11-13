import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x)