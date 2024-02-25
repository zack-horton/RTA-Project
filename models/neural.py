import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import sklearn.metrics as metrics

keras.utils.set_random_seed(2024)

def plot_loss_curves(history, company, model):
    plt.clf()
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title(f"Training and validation loss {company.capitalize()}'s {model}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def get_data(company):
    train = pd.read_parquet(f"data/{company.lower()}/{company.lower()}_train.parquet")
    test = pd.read_parquet(f"data/{company.lower()}/{company.lower()}_test.parquet")

    return train.drop(['price'], axis=1).astype(float).to_numpy(), train[['price']].astype(float).to_numpy(), test.drop(['price'], axis=1).astype(float).to_numpy(), test[['price']].astype(float).to_numpy()

def fit_NN(train_x, train_y, test_x, test_y, company, save_model=False):
    # define the input layer
    input = keras.Input(shape=(76,))

    # feed the long vector to the hidden layer
    x = keras.layers.Dense(76, activation="relu", name="Hidden")(input)

    # feed the output of the hidden layer to the output layer
    output = keras.layers.Dense(1, activation="linear", name="Output")(x)

    # tell Keras that this (input,output) pair is your model
    model = keras.Model(input, output)
    
    model.compile(loss="mean_absolute_error",
                  optimizer="adam",
                  metrics=["mse", "mae"])
    batch_size = 64
    epochs = 25

    history = model.fit(train_x,
                        train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2)
    plot_loss_curves(history, company, "Neural Network")
    model.evaluate(test_x, test_y)
    
    # y_train_pred = model.predict(train_x)
    y_test_pred = model.predict(test_x)
    
    # Test metrics
    explained_variance = metrics.explained_variance_score(test_y, y_test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_y, y_test_pred) 
    mse = metrics.mean_squared_error(test_y, y_test_pred) 
    median_absolute_error = metrics.median_absolute_error(test_y, y_test_pred)
    r2 = metrics.r2_score(test_y, y_test_pred)

    print()
    print("*"*5, f"{company.upper()} NEURAL NET", "*"*5)
    print('explained_variance: ', round(explained_variance, 4))    
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('Median AE: ', round(median_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse),4 ))
    
    if save_model:
        filename = f'models/{company.lower()}_deep_models/{company.lower()}_neural_net.keras'
        model.save(filename)
    
    return model
    
def fit_DNN(train_x, train_y, test_x, test_y, company, save_model=False):
    # define the input layer
    input = keras.Input(shape=(76,))

    # feed the long vector to the hidden layer
    x = keras.layers.Dense(128, activation="relu", name="Hidden_1")(input)
    x = keras.layers.Dense(64, activation="relu", name="Hidden_2")(x)
    x = keras.layers.Dense(32, activation="relu", name="Hidden_3")(x)
    x = keras.layers.Dense(16, activation="relu", name="Hidden_4")(x)

    # feed the output of the hidden layer to the output layer
    output = keras.layers.Dense(1, activation="linear", name="Output")(x)

    # tell Keras that this (input,output) pair is your model
    model = keras.Model(input, output)
    
    model.compile(loss="mean_absolute_error",
                  optimizer="adam",
                  metrics=["mse", "mae"])
    batch_size = 64
    epochs = 25

    history = model.fit(train_x,
                        train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_split=0.2)
    plot_loss_curves(history, company, "Deep Neural Network")
    model.evaluate(test_x, test_y)
    
    # y_train_pred = model.predict(train_x)
    y_test_pred = model.predict(test_x)
    
    # Test metrics
    explained_variance = metrics.explained_variance_score(test_y, y_test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_y, y_test_pred) 
    mse = metrics.mean_squared_error(test_y, y_test_pred) 
    median_absolute_error = metrics.median_absolute_error(test_y, y_test_pred)
    r2 = metrics.r2_score(test_y, y_test_pred)

    print()
    print("*"*5, f"{company.upper()} DEEP NEURAL NET", "*"*5)
    print('explained_variance: ', round(explained_variance, 4))    
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('Median AE: ', round(median_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse),4 ))
    
    if save_model:
        filename = f'models/{company.lower()}_deep_models/{company.lower()}_deep_neural_net.keras'
        model.save(filename)
    
    return model

def fit_neural(train_x, train_y, test_x, test_y, company, save_model=False):
    train_y = train_y.ravel()
    test_y = test_y.ravel()
    # fit_NN(train_x, train_y, test_x, test_y, company, save_model)
    # fit_DNN(train_x, train_y, test_x, test_y, company, save_model)

lyft_train_x, lyft_train_y, lyft_test_x, lyft_test_y = get_data('lyft')
print("lyft train:", len(lyft_train_x), "lyft test:", len(lyft_test_x))

uber_train_x, uber_train_y, uber_test_x, uber_test_y = get_data('uber')
print("uber train:", len(uber_train_x), "uber test:", len(uber_test_x))

fit_neural(lyft_train_x, lyft_train_y, lyft_test_x, lyft_test_y, "lyft", save_model=True)
fit_neural(uber_train_x, uber_train_y, uber_test_x, uber_test_y, "uber", save_model=True)