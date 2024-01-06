import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from extract_layers import ExctractLayers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import platform

if platform.system() == 'Windows':
   feature_extractor = ExctractLayers('../Models/cnn_V2.h5')
else:
    feature_extractor = ExctractLayers('Models/cnn_V2.h5')

features, labels = feature_extractor.extract()

print('X*X*X*X*X*X*X*X*X*X Training Model X*X*X*X*X*X*X*X*X*X')
X = np.array(features)
y = np.array(labels)

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Done splitting the data')

decision_tree_model = DecisionTreeClassifier()

decision_tree_model.fit(X_train, y_train)

print('Training done, testing model')

y_pred = decision_tree_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

if platform.system() == 'Windows':
    model_filename = '../Models/decision_tree.joblib'
else:
    model_filename = 'Models/decision_tree.joblib'

joblib.dump(decision_tree_model, model_filename, protocol=4)
