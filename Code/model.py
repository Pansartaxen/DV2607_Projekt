import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


main_dir = 'Images/clean'
num_classes = 11

train_datagen = ImageDataGenerator(
    rescale=1./255,       # normalize pixel values to be between 0 and 1
    shear_range=0.2,      # shear transformations
    zoom_range=0.2,       # zoom transformations
    horizontal_flip=True,  # horizontal flips
    validation_split=0.2  # set the validation split percentage
)

train_generator = train_datagen.flow_from_directory(
    main_dir,
    target_size=(128, 128),  # set your target size
    batch_size=32,         # set your batch size
    class_mode='categorical',  # use 'categorical' for multi-class classification
    subset='training'  # specify that this is the training set
)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    main_dir,
    target_size=(128, 128),  # set your target size
    batch_size=32,         # set your batch size
    class_mode='categorical',  # use 'categorical' for multi-class classification
    subset='validation',  # specify that this is the validation set
)

# Build the CNN model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, epochs=10)

model.save('Models/cnn.h5')