import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import load_img, img_to_array, to_categorical
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, RandomFlip, RandomRotation
from tensorflow.keras.layers import RandomTranslation, RandomZoom
from tensorflow.keras.layers import Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix
print("Name: Balaji.S")
print("Reg.No: 814724243022")
def load_images_from_path(path, label):
images, labels = [], []
for file in os.listdir(path):
img = load_img(os.path.join(path, file), target_size=(224, 224))
images.append(img_to_array(img))
labels.append(label)
return images, labels
def show_images(images):
fig, axes = plt.subplots(1, min(8, len(images)), figsize=(20, 5))
for i, ax in enumerate(axes):
ax.imshow(images[i].astype("uint8"))
ax.axis("off")

plt.show()
dataset = "/content/arctic-wildlife"
x_train, y_train = [], []
x_test, y_test = [], []
for folder, label in [("arctic_fox",0), ("polar_bear",1), ("walrus",2)]:
imgs, labs = load_images_from_path(f"{dataset}/train/{folder}", label)
x_train += imgs
y_train += labs
for folder, label in [("arctic_fox",0), ("polar_bear",1), ("walrus",2)]:
imgs, labs = load_images_from_path(f"{dataset}/test/{folder}", label)
x_test += imgs
y_test += labs
x_train = preprocess_input(np.array(x_train))
x_test = preprocess_input(np.array(x_test))
y_train = to_categorical(y_train, num_classes=3)
y_test = to_categorical(y_test, num_classes=3)
base_model = ResNet50V2(weights="imagenet", include_top=False,
input_shape=(224,224,3))
for layer in base_model.layers:
layer.trainable = False
model = Sequential([
Rescaling(1./255),
RandomFlip("horizontal"),
RandomTranslation(0.2,0.2),
RandomRotation(0.2),
RandomZoom(0.2),

base_model,
Flatten(),
Dense(1024, activation="relu"),
Dropout(0.2),
Dense(3, activation="softmax")
])
model.compile(
optimizer="adam",
loss="categorical_crossentropy",
metrics=["accuracy"]
)
history = model.fit(
x_train, y_train,
validation_data=(x_test, y_test),
batch_size=10,
epochs=25
)
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
class_labels = ["arctic fox", "polar bear", "walrus"]
y_pred = model.predict(x_test)

cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
xticklabels=class_labels,
yticklabels=class_labels)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()
img = load_img(f"{dataset}/samples/arctic_fox/arctic_fox_140.jpeg",
target_size=(224,224))
plt.imshow(img)
plt.axis("off")
plt.show()
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
prediction = model.predict(x)
for i, label in enumerate(class_labels):
print(f"{label}: {prediction[0][i]:.4f}")
print("Predicted Class:", class_labels[np.argmax(prediction)])
