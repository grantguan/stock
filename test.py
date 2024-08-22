import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.math import argmax, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CLASSES = ["Bull", "Bear"]
LABEL_BULL = CLASSES.index("Bull")
LABEL_BEAR = CLASSES.index("Bear")
datasets = np.load("datasets.npz")
x_train, y_train = datasets["x_train"], datasets["y_train"]
x_val, y_val = datasets["x_val"], datasets["y_val"]
x_test, y_test = datasets["x_test"], datasets["y_test"]

label_distribution = pd.DataFrame([{
    "Dataset": "train",
    "Bull": np.count_nonzero(y_train == LABEL_BULL),
    "Bear": np.count_nonzero(y_train == LABEL_BEAR)},{
    "Dataset": "val",
    "Bull": np.count_nonzero(y_val == LABEL_BULL),
    "Bear": np.count_nonzero(y_val == LABEL_BEAR)},{
    "Dataset": "test",
    "Bull": np.count_nonzero(y_test == LABEL_BULL),
    "Bear": np.count_nonzero(y_test == LABEL_BEAR)},
])
print(len(x_test), len(y_test))
model = keras.models.load_model("best_model.hdf5")
model.evaluate(x_test, to_categorical(y_test))


y_pred_prob = model.predict(x_test)
y_pred = argmax(y_pred_prob, axis=-1)
cm = confusion_matrix(y_test, y_pred, num_classes=len(CLASSES)).numpy()


plt.figure(figsize=(5, 4))
sns.heatmap(cm, xticklabels=CLASSES, yticklabels=CLASSES, annot=True, fmt='g')
plt.xlabel("Prediction")
plt.ylabel("Label")