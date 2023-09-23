"""
===============================
Recognizing hand-written digits
===============================

This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

"""

## Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
## License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt
import cv2

# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split
from utils import data_preprocess, train_model, read_digits, split_train_dev_test, p_and_eval,get_all_h_param_comb,tune_hparams
import pdb

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.


# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, label in zip(axes, digits.images, digits.target):
#     ax.set_axis_off()
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title("Training: %i" % label)

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

all_combos = get_all_h_param_comb(gamma_list,c_list)

h_metric = metrics.accuracy_score

# Load the digits dataset
digits = datasets.load_digits()

# Print the number of total samples in the dataset
total_samples = len(digits.images)
print(f"Number of total samples in the dataset: {total_samples}")

# Get the height and width of the images in the dataset
image_height, image_width = digits.images[0].shape
print(f"Size (height x width) of the images in the dataset: {image_height} x {image_width}")

## Split data 
X, y = read_digits()
# X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.3)
# X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=0.3, dev_size=0.3)

## Use the preprocessed datas
# X_train = data_preprocess(X_train)
# X_dev = data_preprocess(X_dev)
# X_test = data_preprocess(X_test)

# model = train_model(X_train, y_train, {'gamma': 0.001}, model_type='svm')

# Predict the value of the digit on the test subset
# predicted = model.predict(X_test)
# Predict the value of the digit on the test subset
# predicted = p_and_eval(model, X_test, y_test)
###############################################################################
# Below we visualize the first 4 test samples and show their predicted
# digit value in the title.

# _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
# for ax, image, prediction in zip(axes, X_test, predicted):
#     ax.set_axis_off()
#     image = image.reshape(8, 8)
#     ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
#     ax.set_title(f"Prediction: {prediction}")

###############################################################################
# :func:`~sklearn.metrics.classification_report` builds a text report showing
# the main classification metrics.
# print(
#     f"Classification report for classifier {model}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )

# ###############################################################################
# # We can also plot a :ref:`confusion matrix <confusion_matrix>` of the
# # true digit values and the predicted digit values.

# disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
# disp.figure_.suptitle("Confusion Matrix")
# print(f"Confusion matrix:\n{disp.confusion_matrix}")

# plt.show()

###############################################################################
# If the results from evaluating a classifier are stored in the form of a
# :ref:`confusion matrix <confusion_matrix>` and not in terms of `y_true` and
# `y_pred`, one can still build a :func:`~sklearn.metrics.classification_report`
# as follows:


# The ground truth and predicted lists
# y_true = []
# y_pred = []
# cm = disp.confusion_matrix

# For each cell in the confusion matrix, add the corresponding ground truths
# and predictions to the lists
# for gt in range(len(cm)):
#     for pred in range(len(cm)):
#         y_true += [gt] * cm[gt][pred]
#         y_pred += [pred] * cm[gt][pred]

# print(
#     "Classification report rebuilt from confusion matrix:\n"
#     f"{metrics.classification_report(y_true, y_pred)}\n"
# )

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]
image_sizes = [4, 6, 8]


"""
for test_s in test_sizes:
    for dev_s in dev_sizes:
        train_size = 1 - test_s - dev_s
        X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X, y, test_size=test_s, dev_size=dev_s)

        X_train = data_preprocess(X_train)
        X_dev = data_preprocess(X_dev)
        X_test = data_preprocess(X_test)
        
        best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, all_combos ,h_metric)
        
        print(f"test_size={test_s} dev_size={dev_s} train_size={train_size} train_acc={best_accuracy:.2f} dev_acc={best_accuracy:.2f} test_acc={best_accuracy:.2f}")
        print(f"Best Hyperparameters: ( gamma : {best_hparams[0]} , C : {best_hparams[1]} )")
""" 

for image_size in image_sizes:
    # Resize the images to the specified size
    X_resized = [cv2.resize(image, (image_size, image_size)) for image in X]

    # Split data
    X_train, X_test, X_dev, y_train, y_test, y_dev = split_train_dev_test(X_resized, y, test_size=0.2, dev_size=0.1)

    # Preprocess the data
    X_train = data_preprocess(X_train)
    X_dev = data_preprocess(X_dev)
    X_test = data_preprocess(X_test)

    # Tune hyperparameters and train the model
    best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, all_combos, h_metric)

    # Evaluate the model on train, dev, and test sets
    train_predictions = p_and_eval(best_model, X_train, y_train)
    dev_predictions = p_and_eval(best_model, X_dev, y_dev)
    test_predictions = p_and_eval(best_model, X_test, y_test)

    # Calculate accuracies
    train_accuracy = h_metric(train_predictions, y_train)
    dev_accuracy = h_metric(dev_predictions, y_dev)
    test_accuracy = h_metric(test_predictions, y_test)

    # Print the results
    print(f"image size: {image_size}x{image_size} train_size: 0.7 dev_size: 0.1 test_size: 0.2 train_acc: {train_accuracy:.2f} dev_acc: {dev_accuracy:.2f} test_acc: {test_accuracy:.2f}")
    print(f"Best Hyperparameters: (gamma: {best_hparams[0]}, C: {best_hparams[1]})")