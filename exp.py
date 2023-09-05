import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import metrics, svm
from utils import preprocess_digits, train_model, split_train_dev_test,read_digits,predict_and_eval
gamma_ranges = [0.001,0.01,0.1,1,10,100]
C_ranges = [0.1,1,2,3,5,10]

#Dataset
X,y=read_digits()

#Data splitting : train,test,dev
X_train,X_test,X_dev,y_train,y_test,y_dev = split_train_dev_test(X, y, test_size=0.3, dev_size=0.2)

#data preprocessing
X_train=preprocess_digits(X_train)
X_dev=preprocess_digits(X_dev)
X_test=preprocess_digits(X_test)

#hyperparameter tuning
#take all combinations of gamma and c
best_acc = -1
best_model = None
for curr_gamma in gamma_ranges:
    for curr_C in C_ranges:
        #train model with curr gamma and curr C
        curr_model = train_model(X_train,y_train,{'gamma':curr_gamma,'C' : curr_C},model_type="svm")
        #get some performance metric on dev set
        curr_accuracy,predicted=predict_and_eval(curr_model,X_dev,y_dev)
        #select the hparams that give best perf on dev set
        if curr_accuracy>best_acc:
            print(f"New best accuracy {curr_accuracy}")
            best_acc=curr_accuracy
            optimal_gamma=curr_gamma
            optimal_C=curr_C
            best_model = curr_model
print(f"Optimal C is {optimal_C} , Optimal gamma is {optimal_gamma}")


# Predict the value of the digit on the test subset
# 6.Predict and Evaluate 
accuracy,predicted = predict_and_eval(best_model, X_test, y_test)





