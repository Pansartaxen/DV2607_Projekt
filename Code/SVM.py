import joblib
import numpy as np
from sklearn.svm import SVC
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

print('Feature vector shape: ', X.shape)
print('Label vector shape ', y.shape)

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Done splitting the data')

svm_model = SVC(kernel='linear', C=0.5)

print('Fiting the model')

svm_model.fit(X_train, y_train)

print('Training done, testing model')

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

if platform.system() == 'Windows':
    model_filename = '../Models/svm_model.joblib'
else:
    model_filename = 'Models/svm_model.joblib'
joblib.dump(svm_model, model_filename)
