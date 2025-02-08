import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os

MODEL_PATH = "optimized_image_classifier.h5"
TEST_DIR = r"C:\Users\rvhun\Downloads\vista-25\dataset_1\dataset_1\test"
OUTPUT_CSV = "solution.csv"

IMG_SIZE = (224, 224)
BATCH_SIZE = 1

model = tf.keras.models.load_model(MODEL_PATH)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False
)

predictions = model.predict(test_generator)

image_ids = [os.path.basename(path) for path in test_generator.filenames]

df = pd.DataFrame({
    "image_id": image_ids,
    "label": predictions.flatten()
})

df.to_csv(OUTPUT_CSV, index=False)

print(f"Predictions saved to {OUTPUT_CSV}")
