import os
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model

class ExctractLayers:
    '''
    When iniyilizing the class provide
    the path for the model to extract
    features from
    '''
    def __init__(self, Path:str) -> None:
        self.model = load_model(Path)
        self.labels = []
        self.features = []

    def extract(self) -> tuple:
        '''
        Input: None
        Output: Two lists,
            1. Features
            2. Labels
        '''
        # input_layer = self.model.get_layer('conv2d').input
        # output_layer = self.model.get_layer('conv2d_1').output

        input_layer = self.model.get_layer('conv2d').input
        output_layer = self.model.get_layer('dense').output
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

                    feature = feature_extractor_model.predict(img_array)

                    flattened_feature = np.array(feature).flatten()

                    self.labels.append(attacked_img)
                    self.features.append(flattened_feature)

                print(f'X*X*X*X*X*X*X*X*X*X {attack_level} done X*X*X*X*X*X*X*X*X*X')
        return self.features, self.labels

    def save_to_csv(self, path='Models/features.csv') -> None:
        '''
        Input: Path to csv file
        Output: None
        '''
        data = pd.DataFrame({'label': self.labels, 'features': self.features})

        csv_filename = path
        data.astype({'label': int, 'features': str}).to_csv(csv_filename, index=False)
        print(f'Data has been saved to {path}')

    def read_from_csv(self, path='Models/features.csv') -> tuple:
        '''
        Input: Path to csv file
        Output: Two lists,
            1. Features
            2. Labels
        '''
        data = pd.read_csv(path)

        features = np.array(data['features'])
        labels = np.array(data['label'])

        return features, labels

if __name__ == '__main__':
    extract_layers = ExctactLayers('Models/cnn_V2.h5')
    extract_layers.extract()
    print('Extraction done')
    extract_layers.save_to_csv()
