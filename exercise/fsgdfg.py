import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
df = pd.read_csv('titanic.csv')
print(df)
categorical_features = ['Sex', 'Embarked', 'Pclass']
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
X = ct.fit_transform(df)
X = np.array(X)
le = LabelEncoder()
y = le.fit_transform(df['Survived'])
print("Updated matrix of features: \n", X)
print("Updated dependent variable vector: \n", y)
