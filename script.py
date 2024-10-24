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
from scikeras.wrappers import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error


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
    input = layers.InputLayer(shape=(features.shape[1], ))
    hidden = layers.Dense(64, activation='relu')
    # hidden = layers.Dense(32, activation='relu')
    output = layers.Dense(1)
    model.add(input)
    model.add(hidden)
    model.add(output)
    optimizer = Adam(learning_rate=0.01)
    model.compile(loss='mse', metrics=['mae'], optimizer=optimizer)
    return model



# Grid Search Optimization to find best hyperparameters value
def do_grid_search():
    batch_size = [6, 16, 32, 48, 64]
    epochs = [10, 20, 30, 40, 50]
    model = KerasRegressor(build_fn=create_model)
    param_grid = dict(batch_size=batch_size, epochs=epochs)
    # grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False),return_train_score = True)
    grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring='r2')
    grid_result = grid.fit(features_train, labels_train,  verbose = 0)
    print(grid_result)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    return grid_result.best_params_ 



# Create plot
def create_plot(history):
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


# Evaluate model using best params
def evaluate_model():
    model = create_model()
    best_params = do_grid_search()
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)
    history = model.fit(features_train, labels_train, validation_split=0.20, epochs=best_params['epochs'], batch_size=best_params['batch_size'], callbacks=[early_stopping], verbose=0)
    res_mse, res_mae = model.evaluate(features_test, labels_test, verbose=0)
    print(f"MSE: {res_mse}\nMAE: {res_mae}")
    
    # An R Squared Value close to 1 is better
    pred_values = model.predict(features_test)
    r2 = r2_score(labels_test, pred_values)
    print(f"R Squared Value: {r2}")
    
    create_plot(history)
    
    
evaluate_model()

