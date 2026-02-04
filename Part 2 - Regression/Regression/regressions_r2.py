from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
class Regressions:
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.my_dict = {r2: name for r2, name in [
            self.multi_line_reg(),
            self.poly_line_reg(),
            self.deci_tree(),
            self.random_forest()]}
    def multi_line_reg(self):
        regressor = LinearRegression()
        regressor.fit(self.x_train, self.y_train)
        y_pred = regressor.predict(self.x_test)
        return r2_score(self.y_test, y_pred), 'Multiple Linear Regression'
    def poly_line_reg(self):
        poly_reg = PolynomialFeatures(4)
        x_poly = poly_reg.fit_transform(self.x_train)
        regressor = LinearRegression()
        regressor.fit(x_poly, self.y_train)
        y_pred = regressor.predict(poly_reg.transform(self.x_test))
        return r2_score(self.y_test, y_pred), 'Polynomial Regression'
    def deci_tree(self):
        regressor = DecisionTreeRegressor(random_state=0)
        regressor.fit(self.x_train, self.y_train)
        y_pred = regressor.predict(self.x_test)
        return r2_score(self.y_test, y_pred), "Decision Tree Regression"
    def random_forest(self):
        regressor = RandomForestRegressor(random_state=0)
        regressor.fit(self.x_train, self.y_train)
        y_pred = regressor.predict(self.x_test)
        return r2_score(self.y_test, y_pred), "Random Forest Regression"
    def sv_regression(self):
        regressor = SVR()
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        regressor.fit(sc_x.fit_transform(self.x_train), sc_y.fit_transform(self.y_train.reshape(-1, 1)).ravel())
        y_pred = regressor.predict(sc_x.transform(self.x_test))
        return r2_score(self.y_test, sc_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()), "SVR"
    def best_model(self):
        my_dict = self.my_dict
        return max(my_dict), my_dict[max(my_dict)]
    def regressors(self):
        return self.my_dict
dataset = pd.read_csv('Data.csv')
x = dataset.drop(columns='PE').to_numpy()
y = dataset.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=0)
regressions = Regressions(x_train, x_test, y_train, y_test)
print(regressions.sv_regression())