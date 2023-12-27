import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model

# Load the saved HSRRS image classifier model with the custom InputLayer
hsrrs_model = load_model('Models/cnn.h5')

hsrrs_model.summary()

input_layer = hsrrs_model.get_layer('conv2d').input
output_layer = hsrrs_model.get_layer('conv2d_1').output

feature_extractor_model = Model(inputs=input_layer, outputs=output_layer)

parent_dir = "Images"
subdirectories = os.listdir(parent_dir)

class_features = {}

for subdirectory in subdirectories:
    if subdirectory == '.DS_Store':
        continue

    attacked_img = (subdirectory != 'clean')
    class_dir = Path(parent_dir, subdirectory)

    if class_dir.is_file():
        continue

    features_list = []

    for attack_level in class_dir.iterdir():
        if attack_level.name == '.DS_Store':
            continue

        #sub_class = Path(class_dir, attack_level)
        sub_class = Path(attack_level)

        if sub_class.is_file():
            continue
        print('Sub_class:', sub_class)
        for filename in sub_class.iterdir():
            if filename.name == '.DS_Store':
                continue

            img_path = filename

            img = load_img(
                img_path, grayscale=False, color_mode='rgb', target_size=None,
                interpolation='nearest')

            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, 0)

            # Extract features from the adversarial examples
            feature = feature_extractor_model.predict(img_array)

            # Concatenate the features from the two layers
            features = tf.concat([feature], axis=-1)

            features = feature_extractor_model.predict(img_array)

            features_list.append(features)

        class_features[subdirectory] = (attacked_img, np.array(features_list))

# Example of using the extracted features
for class_name, (attacked, features) in class_features.items():
    print(f'Class: {class_name}, Attacked: {attacked}')
    print('Features Shape:', features.shape)
