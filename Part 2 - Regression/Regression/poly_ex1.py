from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=.2)
reg_poly = PolynomialFeatures(4)
x_poly = reg_poly.fit_transform(x_train)
regressor = LinearRegression()
regressor.fit(x_poly, y_train)
y_pred = regressor.predict(reg_poly.transform(x_test))
print(r2_score(y_test, y_pred))