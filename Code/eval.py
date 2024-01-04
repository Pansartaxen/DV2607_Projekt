import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.utils import shuffle
from extract_layers import ExctractLayers
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Load the trained SVM model
model_filename = 'Models/svm_model.joblib'
model = joblib.load(model_filename)

# Load test data
feature_extractor = ExctractLayers('Models/cnn_V2.h5')
features, labels = feature_extractor.extract()

# Shuffle and split the test data
features, labels = shuffle(features, labels, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

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
