
# Import datasets, classifiers and performance metrics
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from joblib import dump,load

from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


#read gigits
def read_digits():
    digits = datasets.load_digits()
    x = digits.images
    y = digits.target 
    return x,y

# We will define utils here :
def preprocess_data(data):
    n_samples = len(data)
    data = data.reshape((n_samples, -1))
    data = normalize(data, norm='l2')
    return data

# Split data into 50% train and 50% test subsets

def split_data(X,y,test_size=0.5,random_state=1):
    X_train, X_test, y_train, y_test = train_test_split(
     X, y, test_size=test_size, shuffle=False
    )
    return X_train, X_test, y_train, y_test

# Create a classifier: a support vector classifier
def train_model(X, y, model_params,model_type = 'svm'):
    if model_type == 'svm':
        clf = svm.SVC(**model_params)
    elif model_type == 'tree':
        clf = tree.DecisionTreeClassifier(**model_params)
    elif model_type == 'logistic_regression':
        clf = LogisticRegression(**model_params)
    clf.fit(X, y)
    return clf

def split_train_dev_test(X, y, test_size, dev_size):
    X_train_dev, X_test, y_train_dev, y_test = split_data(X, y, test_size=test_size)
    X_train, X_dev, y_train, y_dev = split_data(X_train_dev, y_train_dev, test_size=dev_size/(1-test_size))
    return X_train, X_test,X_dev, y_train, y_test, y_dev

def predict_and_eval(model, X, y):
    
    predicted = model.predict(X)
    accuracy = accuracy_score(y, predicted)

    return accuracy,predicted

def tune_hparams(X_train, Y_train, X_dev, y_dev, list_of_all_param_combination, model_type='svm'):
    best_accuracy_so_far = -1
    best_model = None
    best_model_path = ""
    best_hparams = None

    for param_combination in list_of_all_param_combination:
        if model_type == 'svm':
            cur_model = train_model(X_train, Y_train, {'gamma': param_combination['gamma'],'C':param_combination['C']}, model_type='svm')
        if model_type == 'tree':
            cur_model = train_model(X_train, Y_train, {'max_depth': param_combination['max_depth']}, model_type='tree')
        if model_type == 'logistic_regression':
            cur_model = train_model(X_train, Y_train, param_combination, model_type='logistic_regression')

        cur_accuracy,_ = predict_and_eval(cur_model, X_dev, y_dev)
        if cur_accuracy > best_accuracy_so_far:
            best_accuracy_so_far = cur_accuracy
            if model_type == 'svm':
                optimal_gamma = param_combination['gamma']
                optimal_C = param_combination['C']
                best_hparams = {'gamma': optimal_gamma,'C':optimal_C}
            if model_type == 'tree':
                optimal_max_depth = param_combination['max_depth']
                best_hparams = {'max_depth': optimal_max_depth}
            if model_type == 'logistic_regression':
                optimal_solver = param_combination['solver']
                best_hparams = {'solver': optimal_solver}
            best_model = cur_model
            best_model_path = "./models/{}".format(model_type)+"_".join(["{}:{}".format(k,v) for k,v in best_hparams.items()])+".joblib"

            best_model = cur_model

    # save the best model
    dump(best_model,best_model_path)
    #best_model_path = 
    return best_hparams, best_model_path, best_accuracy_so_far

roll_no = "m23csa001"
solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']

def evaluate_logistic_regression(X, y, solvers, roll_no):
    performances = {}
    for solver in solvers:
        model = train_model(X, y, {'solver': solver}, model_type='logistic_regression')
        accuracy, _ = predict_and_eval(model, X, y)
        print(f"Performance with solver {solver}: {accuracy}")
        performances[solver] = accuracy
        model_path = f"{roll_no}_lr_{solver}.joblib"
        dump(model, model_path)
    return performances

def cv_logistic_regression(X, y, solvers, cv=5):
    cv_performances = {}
    for solver in solvers:
        model = LogisticRegression(solver=solver)
        scores = cross_val_score(model, X, y, cv=cv)
        mean_score = scores.mean()
        std_score = scores.std()
        cv_performances[solver] = (mean_score, std_score)
        print(f"Solver: {solver}, Mean CV Score: {mean_score}, Std: {std_score}")
    return cv_performances