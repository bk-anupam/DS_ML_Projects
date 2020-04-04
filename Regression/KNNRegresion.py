import numpy as np
import pandas as pd
import os
from IPython.display import display

dtype_dict = {'bathrooms': float, 'waterfront': int, 'sqft_above': int, 'sqft_living15': float,
              'grade': int, 'yr_renovated': int, 'price': float, 'bedrooms': float, 'zipcode': str,
              'long': float, 'sqft_lot15': float, 'sqft_living': float, 'floors': float, 'condition': int,
              'lat': float, 'date': str, 'sqft_basement': int, 'yr_built': int, 'id': str,
              'sqft_lot': int, 'view': int}


def add_one_vector(X):
    one_vector = np.ones(len(X)).reshape(len(X), 1)
    return np.concatenate((one_vector, X), axis=1)


def load_and_extact(file_name, datatype_dict):
    """
    Load csv data into a dataframe and return a tuple of input features matrix and output feature vector
    :param file_name: name of csv file
    :param datatype_dict: dictionary with dataset column names and their corresponding data types
    :return: a tuple of input features matrix and output feature vector
    """
    sales = pd.read_csv(file_name, dtype=datatype_dict)
    # remove the columns that are not numeric i.e. int, floats etc.
    # sales.dtypes gives a pandas with index as column names and value as column data types. We filter this
    # series to remove columns of type object
    numeric_cols = pd.Series(sales.dtypes).where(lambda col_dtype: col_dtype != 'object').dropna()
    feature_names = list(numeric_cols.keys().values)
    # price is the output variable
    feature_names.remove('price')
    # extract the input features from the dataframe as a numpy 2d array
    input_features = add_one_vector(sales[feature_names].values)
    output_variable = sales.loc[:, 'price'].values
    return input_features, output_variable


def normalize_features(input_features):
    norm = np.sqrt(np.sum(input_features**2, axis=0))
    normalized_features = input_features / norm
    return normalized_features, norm


os.chdir("./data/knn")
train_input_features, train_output_variable = load_and_extact('kc_house_data_small_train.csv', dtype_dict)
print('No of training examples: {}'.format(len(train_input_features)))
cv_input_features, cv_output_variable = load_and_extact('kc_house_data_validation.csv', dtype_dict)
test_input_features, test_output_variable = load_and_extact('kc_house_data_small_test.csv', dtype_dict)
norm_train_input_features, train_norm = normalize_features(train_input_features)
norm_cv_input_features = cv_input_features / train_norm
norm_test_input_features = test_input_features / train_norm


def get_min_distance_row_index(rowindex_distance):
    rowindex = rowindex_distance[:, 0]
    distance = rowindex_distance[:, 1]
    min_distance = np.amin(distance)
    min_distance_index = np.where(distance == min_distance)
    return rowindex[min_distance_index], min_distance


def compute_distance_loop(training_examples, query_house):
    knn = []
    i = 0
    for train_house in training_examples:
        knn.append([i, np.sqrt(np.sum((train_house - query_house)**2))])
        i += 1
    return np.array(knn)


def compute_distance_vectorized(training_examples, query_house):
    """
    Vectorized implementation of calculating the distance of a query house from each of the training examples
    :param training_examples: a matrix or numpy 2d array consisting of training data (input features)
    :param query_house:
    :return: numpy 2d array whose first column is the training row index and second column is the distance from
    the query house
    """
    # subtract the query house row from each training example row
    diff_matrix = training_examples - query_house
    # now for each row in the matrix (which corresponds to each training example), calculate the sum of
    # squares of feature values ( this is done by using axis = 1 in the 2d array )
    distance = np.sqrt(np.sum(diff_matrix**2, axis=1))
    index = np.arange(0, len(distance))
    return np.concatenate((index.reshape(-1, 1), distance.reshape(-1, 1)), axis=1)


def compare_distance_implementation():
    rowindex_distance_loop = compute_distance_loop(norm_train_input_features[0:10, :], norm_test_input_features[0])
    rowindex_distance_vectorized = compute_distance_vectorized(norm_train_input_features[0:10, :], norm_test_input_features[0])
    rowindex_loop, min_distance_loop = get_min_distance_row_index(rowindex_distance_loop)
    rowindex_vectorized, min_distance_vectorized = get_min_distance_row_index(rowindex_distance_vectorized)
    print('For loop implementation')
    print(rowindex_distance_loop)
    print('The nearest neighbour is at training example number: {} with min. distance = {}'
          .format(rowindex_loop, min_distance_loop))
    print('For vectorized implementation')
    print(rowindex_distance_vectorized)
    print('The nearest neighbour is at training example number: {} with min. distance = {}'
          .format(rowindex_vectorized, min_distance_vectorized))


def one_nearest_neighbour(training_set, query_house):
    rowindex_distance = compute_distance_vectorized(training_set, query_house)
    distance = rowindex_distance[:, 1]
    print(distance)
    print(min(distance))
    rowindex_min, min_distance = get_min_distance_row_index(rowindex_distance)
    print(rowindex_min, min_distance)

def run_knn():
    print('1st test example:')
    print(norm_test_input_features[0])
    print('\n10th training example:')
    print(norm_train_input_features[9])
    print('\nEucledian distance between 10th training example and 1st test example:')
    print(np.sqrt(np.sum((norm_train_input_features[9] - norm_test_input_features[0]) ** 2)))
    distance = {}
    for i in range(10):
        distance[i] = np.sqrt(np.sum((norm_train_input_features[i] - norm_test_input_features[0]) ** 2))
    print(distance)
    # now compute the distance of the query house ( the first test row ) from each of the training examples
    compare_distance_implementation()
    diff = norm_train_input_features[:] - norm_test_input_features[0]
    print(diff[-1].sum())
    total_row = np.sum(diff ** 2, axis=1)
    print(total_row.shape)
    print(diff.shape)
    print(np.sum(diff**2, axis=1)[15])
    #one_nearest_neighbour(norm_train_input_features, norm_test_input_features[0])


if __name__ == "__main__":
    run_knn()