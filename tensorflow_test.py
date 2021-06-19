import tensorflow as tf
import numpy as np
from pprint import pprint

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(5, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=["accuracy"]
)


def generate_data(n):
    x_train = []
    y_train = []

    for _ in range(n):
        data = np.random.default_rng().standard_normal(20)
        s = sum(data)
        x_train.append(data)
        y_train.append(s < 2)

    return np.array(x_train), np.array(y_train)


x_train, y_train = generate_data(100000)
x_test, y_test = generate_data(1000)

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)
pprint(model.trainable_variables)
# print(model(np.array([[3, 5], [2, -10], [3, 2], [0, 0]])))
