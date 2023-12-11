import cv2
import os

input_dir = "UCMerced_LandUse/Images/old_storagetanks"
output_dir = "UCMerced_LandUse/Images/storagetanks"
desired_width = 128
desired_height = 128

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through the images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".tif"):  # Change the file extension to match your images
        input_path = os.path.join(input_dir,filename)
        img = cv2.imread(input_path)
        if img is not None:
            img_resized = cv2.resize(img, (desired_width, desired_height))
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, img_resized)