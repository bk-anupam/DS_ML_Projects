import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from IPython.display import display


def load_and_extract(file_name):
    sales_df = pd.read_csv(file_name)
    sales_df.sort_values(by=['sqft_living', 'price'], inplace=True)
    print('The top 5 rows of the dataset: {}'.format(file_name))
    display(sales_df.head())
    # The target variable that we want to predict is price and the input features that we consider is just sqft_living
    sqft_living = sales_df.loc[:, 'sqft_living'].values.reshape(-1, 1)
    price = sales_df.loc[:, 'price'].values.reshape(-1, 1)
    return sqft_living, price


# plot the house price versus sqft_living
def plot_sqftliving_price(sqft_living, price):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(sqft_living, price, s=6, color='orange')
    xrange = np.linspace(0, 14000, 15)
    ax.set_xticks(xrange)
    plt.xlabel('sqft_living')
    plt.ylabel('price')
    return fig, ax


def regression_degree_n(degree_n, X, y):
    polynomial_features = PolynomialFeatures(degree=degree_n)
    X_degree_n = polynomial_features.fit_transform(X)
    polynomial_model = LinearRegression()
    polynomial_model_train_degn = polynomial_model.fit(X_degree_n, y)
    y_predicted_degree_n = polynomial_model_train_degn.predict(X_degree_n)
    return y_predicted_degree_n, polynomial_model_train_degn


def plot_model_train_error_vs_degree(degree_arr, model_train_error_arr, model_cv_error_arr):
    fig_err, ax_err = plt.subplots()
    ax_err.plot(degree_arr, model_cv_error_arr, marker='o', label='Validation error')
    ax_err.plot(degree_arr, model_train_error_arr, marker='x', label='Training error')
    plt.xlabel('model complexity (polynomial degree)')
    plt.ylabel('root mean squared error')
    plt.yscale("log")
    plt.legend()
    plt.title("Error vs model complexity")
    plt.show()


def get_cv_error(degree_n, poly_model_train, X_cv, y_cv):
    polynomial_features = PolynomialFeatures(degree=degree_n)
    X_cv_degree_n = polynomial_features.fit_transform(X_cv)
    y_cv_predicted = poly_model_train.predict(X_cv_degree_n)
    return np.sqrt(mean_squared_error(y_cv, y_cv_predicted))


def get_min_cv_error_degree(model_cv_error_2darr):
    degree_arr = model_cv_error_2darr[:, 0]
    cv_err_arr = model_cv_error_2darr[:, 1]
    min_err = np.amin(cv_err_arr)
    min_err_index = np.where(cv_err_arr == min_err)
    min_err_degree = degree_arr[min_err_index]
    return min_err_degree, min_err



def train_models(sqft_living_train, price_train, sqft_living_cv, price_cv):
    # Dictionary mapping polynomial degree to line plot color
    deg_color_map = {1: 'red', 2: 'green', 3: 'blue', 4: 'cyan', 5: 'grey',
                     6: 'gold', 7: 'lavender', 8: 'lime', 9: 'magenta', 10: 'coral'}
    # list of tuples where the
    model_train_error = []
    model_cv_error = []
    # we will train models with polynomial degree 1 to 10 and select the one with the lowest rmse
    for degree in range(10):
        degree = degree + 1
        predicted_price_train_degn, poly_model_train = regression_degree_n(degree, sqft_living_train, price_train)
        # plot the polynomial fit
        plt.plot(sqft_living_train, predicted_price_train_degn, color=deg_color_map.get(degree),
                 label='degree {}'.format(degree))
        rmse_train = np.sqrt(mean_squared_error(price_train, predicted_price_train_degn))
        rmse_cv = get_cv_error(degree, poly_model_train, sqft_living_cv, price_cv)
        model_train_error.append([degree, rmse_train])
        model_cv_error.append([degree, rmse_cv])
    plt.legend()
    plt.title('Polynomial regression')
    model_cv_error_arr = np.array(model_cv_error)
    model_train_error_arr = np.array(model_train_error)
    min_err, min_err_degree = get_min_cv_error_degree(model_cv_error_arr)
    print("The minimum cross validation error is: {} for polynomial degree: {}".format(min_err, min_err_degree[0]))
    plot_model_train_error_vs_degree(model_cv_error_arr[:, 0], model_train_error_arr[:, 1], model_cv_error_arr[:, 1])


def run_poly_regression():
    os.chdir('data')
    sqft_living_all, price_all = load_and_extract('kc_house_data.csv')
    sqft_living_train, price_train = load_and_extract('wk3_kc_house_train_data.csv')
    sqft_living_cv, price_cv = load_and_extract('wk3_kc_house_valid_data.csv')
    sqft_living_test, price_test = load_and_extract('wk3_kc_house_test_data.csv')
    fig, ax = plot_sqftliving_price(sqft_living_train, price_train)
    train_models(sqft_living_train, price_train, sqft_living_cv, price_cv)


if __name__ == "__main__":
    run_poly_regression()
