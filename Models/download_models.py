import os
import gdown

def download_files(file_dict):
    for file_path, drive_link in file_dict.items():
        if not os.path.exists(file_path):
            print(f"Downloading {file_path} from {drive_link}")
            gdown.download(drive_link, file_path, quiet=False)
        else:
            print(f"File {file_path} already exists.")

file_dict = {
    'cnn_V2.h5': 'https://drive.google.com/uc?id=1IjY7UtTB_EoW4vM4QPObnxRWyC2_VV0-',
    'svm.joblib': 'https://drive.google.com/uc?id=1k93jz8ZzlKUkcV4j1R0C15IIPD9xsfbM',
    'knn.joblib': 'https://drive.google.com/uc?id=1D34T1vhp3p2MqrMivDQFUvidMXt1mr9S',
    'decision_tree.joblib': 'https://drive.google.com/uc?id=182Q0iwV99kqWYbrVfnneBNA1IJuJbIS8'
}


download_files(file_dict)
