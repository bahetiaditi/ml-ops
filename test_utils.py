import pytest
import numpy as np
from utils import split_train_dev_test,read_digits,preprocess_data,tune_hparams
import os
from api.app import app
import json
from utils import read_digits
from sklearn.linear_model import LogisticRegression
from joblib import load
import unittest

@pytest.fixture(scope="module")
def client():
    with app.test_client() as client:
        yield client

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

def test_logistic_regression_model_type():
    roll_no = "m23csa001"  
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    for solver in solvers:
        model_path = f"{roll_no}_lr_{solver}.joblib"
        assert os.path.exists(model_path), f"Model file {model_path} does not exist"

        model = load(model_path)
        assert isinstance(model, LogisticRegression), f"Loaded model is not a LogisticRegression model"

def test_logistic_regression_solver_name():
    roll_no = "m23csa001"  
    solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

    for solver in solvers:
        model_path = f"{roll_no}_lr_{solver}.joblib"
        assert os.path.exists(model_path), f"Model file {model_path} does not exist"

        model = load(model_path)
        model_solver = model.get_params()['solver']
        assert model_solver == solver, f"Model solver {model_solver} does not match expected solver {solver}"
        
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

def test_predict_svm(client):
    test_data = {'image': ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]}
    sample = np.array(test_data['image']).reshape(1, -1)
    response = client.post('/predict/svm', json={'image': sample.tolist()})
    assert response.status_code == 200


