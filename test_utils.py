import pytest
import numpy as np
from utils import split_train_dev_test,read_digits,preprocess_data,tune_hparams
import os
from api.app import app
import json
from utils import read_digits

@pytest.fixture(scope="module")
def digit_samples():
    x, y = read_digits()
    samples = {}
    for digit in range(10):
        sample_index = np.where(y == digit)[0][0]
        samples[digit] = x[sample_index].reshape(1, -1)  # Reshape if necessary
    return samples

@pytest.mark.parametrize("digit, expected_prediction", [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (9, 9),
])

def test_post_predict(digit_samples, digit, expected_prediction):
    sample = digit_samples[digit]

    # Sending a POST request to the /predict endpoint
    response = app.test_client().post("/predict", json={"image": sample.tolist()})

    # Check the status code and the prediction
    assert response.status_code == 200
    response_data = json.loads(response.get_data(as_text=True))
    prediction = response_data['prediction'][0]
    assert prediction == expected_prediction


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

