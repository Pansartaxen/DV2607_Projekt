import os
import gdown

def download_files(file_dict):
    for file_path, drive_link in file_dict.items():
        if not os.path.exists(file_path):
            print(f"Downloading {file_path} from {drive_link}")
            gdown.download(drive_link, file_path, quiet=False)
        else:
            print(f"File {file_path} already exists.")

# Example usage
file_dict = {
    '../Models/cnn.h5': 'https://drive.google.com/uc?id=1ZgWZgfcM-cKs4-WOVENZDR1PtECj5z7A',
}

download_files(file_dict)
