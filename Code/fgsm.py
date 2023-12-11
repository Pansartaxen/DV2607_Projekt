import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

PATHS = {
    "airport": "../Images/airport",
    "avenue": "../Images/avenue",
    "bridge": "../Images/bridge",
    "building": "../Images/building",
    "denseresidential": "../Images/denseresidential",
    "highway": "../Images/highway",
    "marina": "../Images/marina",
    "mediumresidential": "../Images/mediumresidential",
    "parkinglot": "../Images/parkinglot",
    "residents": "../Images/residents",
    "storeroom": "../Images/storeroom",
}

def load_image(img_path, size=(128, 128)):
    img = image.load_img(img_path, target_size=size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def fgsm_attack(input_folder, output_folder, epsilon):
    # Load the pre-trained model
    model = load_model('model.h5')
    num_classes = 11
    # Ensure output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for img_name in os.listdir(input_folder):
        img_path = os.path.join(input_folder, img_name)
        img_array = load_image(img_path)

        # Convert the image array to a TensorFlow tensor
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        # Get the input label of the image
        preds = model.predict(img_tensor)
        label_index = np.argmax(preds[0])
        label = tf.one_hot(label_index, num_classes)

        # Reshape the label to match the output shape
        label = tf.reshape(label, (1, num_classes))

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            prediction = model(img_tensor)
            loss = tf.keras.losses.categorical_crossentropy(label, prediction)

        # Get the gradients of the loss w.r.t to the input image
        gradient = tape.gradient(loss, img_tensor)

        # Get the sign of the gradients to create the perturbation
        signed_grad = tf.sign(gradient)

        # Apply FGSM
        adversarial_img = img_tensor + epsilon * signed_grad
        adversarial_img = tf.clip_by_value(adversarial_img, -1, 1)

        # Save the adversarial image
        save_path = os.path.join(output_folder, img_name)
        adversarial_img_squeezed = np.squeeze(adversarial_img.numpy(), axis=0)
        tf.keras.preprocessing.image.save_img(save_path, adversarial_img_squeezed)

# Example usage

#for folders in PATHS:
#    fgsm_attack(input_folder='../Images/airport', output_folder=f'../Images/fgsm/{folders}', model_path='../Models/cnn.h5',
#                epsilon=0.01)

fgsm_attack(input_folder='../Images/airport', output_folder=f'../Images/fgsm/airport',
            epsilon=0.1)

#fgsm_attack(input_folder='../Images/airport', output_folder=f'../Images/fgsm/airport', model_path='../Models/cnn.h5',
#            epsilon=0.01)