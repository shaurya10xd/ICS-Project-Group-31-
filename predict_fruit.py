import json
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

model = load_model(
    "best_vgg16_fruit_model.h5"
)

with open(
    "class_indices.json",
    "r"
) as f:
    class_indices = json.load(f)

class_names = {
    value: key
    for key, value in class_indices.items()
}

img_path = input(
    "Enter image path: "
)

img = image.load_img(
    img_path,
    target_size=(224, 224)
)

img_array = image.img_to_array(
    img
)

img_array = np.expand_dims(
    img_array,
    axis=0
)

img_array = preprocess_input(
    img_array
)

predictions = model.predict(
    img_array,
    verbose=0
)

predicted_index = np.argmax(
    predictions
)

predicted_class = class_names[
    predicted_index
]

confidence = (
    predictions[0][predicted_index] * 100
)

print("\nPrediction")
print("=" * 40)

print(
    f"Fruit: {predicted_class}"
)

print(
    f"Confidence: {confidence:.2f}%"
)

print("\nTop 3 Predictions")
print("=" * 40)

top3 = np.argsort(
    predictions[0]
)[-3:][::-1]

for idx in top3:
    print(
        f"{class_names[idx]} : "
        f"{predictions[0][idx] * 100:.2f}%"
    )