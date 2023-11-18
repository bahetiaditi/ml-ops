import pytest
from utils import split_train_dev_test,read_digits,preprocess_data,tune_hparams
import os

def inc(x):
    return x + 1

def test_inc():
    assert inc(4) == 5

def create_dummy_hyperparamete():
    gama_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    C_ranges = [0.1,1,2,5,10]
    list_of_all_param_combination = [{'gamma': gamma, 'C': C} for gamma in gama_ranges for C in C_ranges]
    return list_of_all_param_combination

def create_dummy_data():
    X,y = read_digits()
    X_train = X[:100,:,:]
    y_train = y[:100]
    X_dev = X[:50,:,:]
    y_dev = y[:50]
    X_train = preprocess_data(X_train)
    X_dev = preprocess_data(X_dev)
    return X_train, y_train, X_dev, y_dev

def test_hparam_count():
     list_of_all_param_combination = create_dummy_hyperparamete()
     assert len(list_of_all_param_combination) == 35


def test_mode_saving():
    X_train, y_train, X_dev, y_dev = create_dummy_data()
    list_of_all_param_combination = create_dummy_hyperparamete()
    _, best_model_path, _ = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination,'svm')
    assert os.path.exists(best_model_path)

def test_data_splitting():
    X,y = read_digits()
    X = X[:100,:,:]
    y = y[:100]

    test_size = 0.1
    dev_size = 0.6
    train_size = 1 - (dev_size + test_size)

    X_train, X_test,X_dev, y_train, y_test,y_dev = split_train_dev_test(X, y, test_size=test_size, dev_size=dev_size);
    assert len(X_train) == int(train_size * len(X)) and len(X_test) == int(test_size * len(X)) and len(X_dev) == int(dev_size * len(X))

