from __future__ import print_function, division

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

IMAGE_SIZE = [224, 224]

epochs = 30
batch_size = 32

train_path = "FruitRecognition/train"
test_path = "FruitRecognition/test"

folders = glob(train_path + "/*")

print("Number of Classes:", len(folders))

if len(folders) == 0:
    raise ValueError(
        f"No class folders found in {train_path}"
    )

vgg = VGG16(
    input_shape=IMAGE_SIZE + [3],
    weights="imagenet",
    include_top=False
)

for layer in vgg.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(vgg.output)

x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)

prediction = Dense(
    len(folders),
    activation="softmax"
)(x)

model = Model(
    inputs=vgg.input,
    outputs=prediction
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True
)

test_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = train_gen.flow_from_directory(
    train_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_gen.flow_from_directory(
    test_path,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

print("\nClass Indices:")
print(train_generator.class_indices)

import json

with open("class_indices.json", "w") as f:
    json.dump(
        train_generator.class_indices,
        f,
        indent=4
    )

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "best_vgg16_fruit_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=epochs,
    callbacks=[early_stop, checkpoint]
)

print("\nGenerating Confusion Matrix...")

predictions = model.predict(
    test_generator,
    verbose=1
)

predictions = np.argmax(
    predictions,
    axis=1
)

true_labels = test_generator.classes

cm = confusion_matrix(
    true_labels,
    predictions
)

print("\nConfusion Matrix:")
print(cm)

model.save(
    "vgg16_fruit_recognition.h5"
)

print("\nModel saved successfully!")

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.savefig("loss_curve.png")
plt.close()

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("accuracy_curve.png")
plt.close()

loss, accuracy = model.evaluate(
    test_generator,
    verbose=1
)

print("\nFinal Test Accuracy:", accuracy)
print("Final Test Loss:", loss)

print("\nTraining Complete!")