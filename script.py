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

model.fit(features_train, labels_train, epochs=40, batch_size=2, verbose=1)

res_mse, res_mae = model.evaluate(features_test, labels_test, verbose=0)


# print(model.summary()) 
# print(model.weights)
# print(features_train)
# print(features_test)
# print(labels_train)
# print(labels_test)
print(f"mse: {res_mse}\nmae: {res_mae}")