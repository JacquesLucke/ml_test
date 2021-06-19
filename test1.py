import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10),
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# plt.imshow(x_train[0])
# plt.show()
predictions = model(x_train[:1]).numpy()
probabilities = tf.nn.softmax(predictions).numpy()
print(predictions)
print(probabilities)
print(y_train[0])
print(loss_fn(y_train[:1], predictions).numpy())

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test, verbose=2)

correct_count = 0
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
for test, expected in zip(x_test[:1000], y_test[:1000]):
    probabilities = probability_model(np.expand_dims(test, axis=0))[0]
    result = np.argmax(probabilities)
    correct_count += expected == result

print(correct_count)
