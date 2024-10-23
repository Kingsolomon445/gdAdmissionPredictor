import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score


df = pd.read_csv("admissions_data.csv")

# Divide into features and labels
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]


# Remove serial no
features = features.drop(columns=['Serial No.'])

# Splitting into training and test set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels) 


# Tranform data to use similar scale
normalizer = Normalizer().fit(features_train)
features_train = normalizer.transform(features_train)
features_test = normalizer.transform(features_test)


def create_model():
    model = Sequential()
    input = layers.InputLayer(input_shape=(features.shape[1], ))
    hidden = layers.Dense(64, activation='relu')
    # hidden = layers.Dense(32, activation='relu')
    output = layers.Dense(1)
    model.add(input)
    model.add(hidden)
    model.add(output)
    return model


model = create_model()
optimizer = Adam(learning_rate=0.01)
model.compile(loss='mse', metrics=['mae'], optimizer=optimizer)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)


history = model.fit(features_train, labels_train, validation_split=0.20, epochs=15, batch_size=10, callbacks=[early_stopping], verbose=0)
# history = model.fit(features_train, labels_train, validation_split=0.2, epochs=40, batch_size=2, verbose=0)

res_mse, res_mae = model.evaluate(features_test, labels_test, verbose=0)
pred_values = model.predict(features_test)
# pred_values = pred_values.flatten()
r2 = r2_score(labels_test, pred_values)

print(f"MSE: {res_mse}\MAE: {res_mae}")
print(f"R Squared Value: {r2}")


# Create plot
fig = plt.figure()

# Subplot for loss
plt.subplot(2, 1, 1)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(['Train', 'Val'], loc='upper left')

# Subplot for MAE
plt.subplot(2, 1, 2)
plt.plot(history.history["mae"])
plt.plot(history.history["val_mae"])
plt.title("Mean Average Error")
plt.ylabel("MAE")
plt.xlabel("Epoch")
plt.legend(['Train', 'Val'], loc='upper left')
 
plt.tight_layout() # prevents plots from overlapping each other 
plt.show()




# print(model.summary()) 
# print(model.weights)
# print(features_train)
# print(features_test)
# print(labels_train)
# print(labels_test)
# print(history.history)

