import tensorflow as tf
import random
import numpy as np
from PIL import Image

network = tf.keras.models.load_model("NeuralNetwork.h5")

data = tf.keras.datasets.mnist.load_data()
test = data[1]
test_examples, test_labels = test

test_reshaped = test_examples.reshape((10000, 28, 28, 1))
test_reshaped = tf.dtypes.cast(test_reshaped, dtype=tf.float32)

predictions = network.predict(test_reshaped)
predictions_argmax = np.argmax(predictions, axis=-1)

while True:
    rand_idx = random.randint(0, 10000-1)
    example = test_examples[rand_idx]
    label = test_labels[rand_idx]
    prediction = predictions_argmax[rand_idx]
    img = Image.fromarray(example, mode='L')
    if not prediction == label:
        break

print('Ex #', rand_idx)
print('prediction:', prediction) 
print('Label: ', label)
img.resize((252, 252)).show()

