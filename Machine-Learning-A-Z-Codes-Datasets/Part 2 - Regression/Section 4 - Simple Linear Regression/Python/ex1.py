import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = pd.read_csv('Salary_Data.csv')
x = df[[df.columns[0]]].to_numpy()
y = df.iloc[:, 1].to_numpy()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
# print(x_test)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.show()
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.show()