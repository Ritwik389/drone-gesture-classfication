import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIG ---
DATA_DIR = "dataset"
IMG_SIZE = 128
BATCH_SIZE = 32
MODEL_PATH = "drone_cnn.keras"

print("Loading Data & Model...")
# We use the EXACT same scaling as training (1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load data without shuffling so we can match predictions to labels
test_generator = test_datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False 
)

model = tf.keras.models.load_model(MODEL_PATH)

# --- PREDICT ---
print("Running Predictions (this takes a moment)...")
Y_pred = model.predict(test_generator)
y_pred = np.argmax(Y_pred, axis=1)

# --- REPORT ---
class_labels = list(test_generator.class_indices.keys())

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(test_generator.classes, y_pred, target_names=class_labels))

print("\n--- CONFUSION MATRIX ---")
cm = confusion_matrix(test_generator.classes, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.title('Where is the Model Confused?')
plt.show()