import os
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model

class ExctactLayers:
    '''
    When iniyilizing the class provide
    the path for the model to extract
    features from
    '''
    def __init__(self, Path) -> None:
        self.model = load_model(Path)
        self.labels = []
        self.features = []

    def extract(self) -> list:
        '''
        Input: None
        Output: Two lists,
            1. Features
            2. Labels
        '''
        input_layer = self.model.get_layer('conv2d').input
        output_layer = self.model.get_layer('conv2d_1').output

        feature_extractor_model = Model(inputs=input_layer, outputs=output_layer)

        parent_dir = 'Images'
        subdirectories = os.listdir(parent_dir)


        for subdirectory in subdirectories:
            if subdirectory == '.DS_Store':
                continue

            attacked_img = (subdirectory != 'clean')
            class_dir = Path(parent_dir, subdirectory)

            if class_dir.is_file():
                continue

            for attack_level in class_dir.iterdir():
                if attack_level.name == '.DS_Store':
                    continue

                sub_class = Path(attack_level)

                if sub_class.is_file():
                    continue

                for filename in sub_class.iterdir():
                    # Fail safe for Mac
                    if filename.name == '.DS_Store':
                        continue

                    img_path = filename

                    img = load_img(
                        img_path,
                        grayscale=False,
                        color_mode='rgb',
                        target_size=None,
                        interpolation='nearest'
                    )

                    img_array = img_to_array(img)
                    img_array = np.expand_dims(img_array, 0)

                    # Extract self.features from the adversarial and clean examples
                    feature = feature_extractor_model.predict(img_array)

                    flattened_feature = np.array(feature).flatten()

                    self.labels.append(attacked_img)
                    self.features.append(flattened_feature)
                    # Concatenate the self.features from the layers

                    #df_subdirectory = pd.concat([df_subdirectory, pd.DataFrame({'label': [attacked_img], 'self.features': [flattened_feature.tolist()]})], ignore_index=True)
                print(f'X*X*X*X*X*X*X*X*X*X {attack_level} done X*X*X*X*X*X*X*X*X*X')
        return self.features, self.labels

    def save_to_csv(self, path='Models/features.csv') -> None:
        data = pd.DataFrame({'label': self.labels, 'features': self.features})

        csv_filename = path
        data.astype({'label': int, 'features': str}).to_csv(csv_filename, index=False)