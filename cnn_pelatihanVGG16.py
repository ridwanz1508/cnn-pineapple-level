import tensorflow as tf
from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

# Path to dataset
dataset_dir = "C:\\Users\\hp\\00-Python\\CnnProject\\dataset_nanas"

# Number of classes
num_classes = 2

# Load VGG16 model (without classification top)
base_model = VGG16(weights="imagenet", include_top=False)

# Add global average pooling and new classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)

# Create model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Data preprocessing settings
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Load training data
train_generator = train_datagen.flow_from_directory(
    dataset_dir + "/train",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

# Load validation data
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
validation_generator = validation_datagen.flow_from_directory(
    dataset_dir + "/validation",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
)

# Train model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10,
)

# Save trained model to h5 file
model.save("trained_vgg16_model.h5")

# Load test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    dataset_dir + "/test",
    target_size=(224, 224),
    batch_size=1,
    class_mode="categorical",
    shuffle=False,
)

# Evaluate model
accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {accuracy[1]}")
