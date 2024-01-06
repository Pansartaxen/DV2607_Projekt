import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils import shuffle
from extract_layers import ExctractLayers
from sklearn.model_selection import train_test_split
import platform

if platform.system() == 'Windows':
   feature_extractor = ExctractLayers('../Models/cnn_V2.h5')
else:
    feature_extractor = ExctractLayers('Models/cnn_V2.h5')

X, y = feature_extractor.extract()

# Shuffle and split the test data
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if platform.system() == 'Windows':
    FILES = ['../Models/decision_tree.joblib', '../Models/knn.joblib', '../Models/svm_model_v2.joblib']
else:
    FILES = ['Models/decision_tree.joblib', 'Models/knn.joblib', 'Models/svm_model_v2.joblib']
for file in FILES:
    print(f'X*X*X Evaluating {file[7:]} X*X*X')
    model = joblib.load(file)
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Calculate rates
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TP = conf_matrix[1][1]
    TN = conf_matrix[0][0]

    # Print the confusion matrix and rates
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nFalse Positive (FP):", FP)
    print("False Negative (FN):", FN)
    print("True Positive (TP):", TP)
    print("True Negative (TN):", TN)

    # Calculate and print accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", accuracy)

    # Print classification report for additional metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot the confusion matrix as a heatmap
    labels = ['Negative', 'Positive']
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
