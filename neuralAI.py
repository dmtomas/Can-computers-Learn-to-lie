import csv
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
import time


def train_Neural(c):
    # Train with the data given by the previous games.
    name = "Training_and_graph_data/neural_network_data" + str(c) + ".csv"
    with open(name) as csvfile:
        reader = csv.reader(csvfile)
        next(reader)

        data = []
        for row in reader:
            data.append({
                "evidence": [float(cell) for cell in row[:6]],
                "label": [1, 0, 0, 0] if row[6] == "0" else [0, 1, 0, 0] if row[6] == "1" else [0, 0, 1, 0] if row[6] == "2" else [0, 0, 0, 1]
            })

    # Separate data into training and testing groups
    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]
    X_training, X_testing, y_training, y_testing = train_test_split(
        evidence, labels, test_size=0.1
    )
    try:
        model = tf.keras.models.load_model('saved_model/my_model2')
    except:
        # Create a neural network
        model = tf.keras.models.Sequential()

        # Add two hidden layers with 4 units, with ReLU activation
        model.add(tf.keras.layers.Dense(6, input_shape=(6,), activation="relu"))
        model.add(tf.keras.layers.Dense(6, input_shape=(6,), activation="relu"))

        # Add output layer with 1 unit, with sigmoid activation
        model.add(tf.keras.layers.Dense(4, activation="sigmoid"))

        # Train neural network
        model.compile(
            optimizer="adam",
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    # How many times it has to go throw the data.
    model.fit(X_training, y_training, epochs=1)

    # Evaluate how well model performs
    model.evaluate(X_testing, y_testing, verbose=0)

    model.save('saved_model/my_model2')


# This way the code only runs if you run the full script.
if __name__ == '__main__':
    # for i in range(0, 51):
    #    print("The AI was trained with: " + str(i) + " documents.")
    #    train_Neural(i)

    new_model = tf.keras.models.load_model('saved_model/my_model')
    x = tf.Variable([np.array([0, 0, 12, 0, 4, 0])], trainable=False, dtype=tf.float64)

    print(new_model(x)[0])

