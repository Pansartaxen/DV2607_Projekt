import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import shuffle
from extract_layers import ExctactLayers
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

feature_extractor = ExctactLayers('Models/cnn_V2.h5')

features, labels = feature_extractor.extract()

print('X*X*X*X*X*X*X*X*X*X Training Model X*X*X*X*X*X*X*X*X*X')
X = np.array(features)
y = np.array(labels)

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='linear', C=1.0)

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

model_filename = 'Models/svm_model.joblib'
joblib.dump(svm_model, model_filename)
