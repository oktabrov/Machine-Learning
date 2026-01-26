import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
dt = pd.read_csv('titanic.csv')
ct = ColumnTransformer(transformers = [('encode', OneHotEncoder(), ['Sex', 'Embarked', 'Pclass'])], remainder = 'passthrough')
X = np.array(ct.fit_transform(dt))
print(X)
le = LabelEncoder()
Y = le.fit_transform(dt['Survived'])
print(Y)