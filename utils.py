import matplotlib.pyplot as plt
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


def read_digits():
    digits=datasets.load_digits()
    x = digits.images
    y = digits.target 
    return x,y



def preprocess_digits(dataset):
    n_samples = len(dataset)
    data = dataset.reshape((n_samples, -1))
    return data

def split_data(X,y,test_size=0.5,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test


def split_train_dev_test(X, y, test_size, dev_size):
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    X_train, X_dev, y_train, y_dev = split_data(X_train, y_train, test_size=dev_size)
    return X_train, X_test,X_dev, y_train, y_test, y_dev

def train_model(x,y,model_params,model_type="svm"):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    clf.fit(X, y)
    return clf


def predict_and_eval(model, X_test, y_test):
    # Getting model predictions on the test set
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test,predicted),predicted

    