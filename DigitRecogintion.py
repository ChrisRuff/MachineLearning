#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import random
import tensorflow as tf
import numpy as np    #CPP WRAPER TO PARSE MATRIX'S
from PIL import Image #IMAGE VIEWER


# In[2]:


data = tf.keras.datasets.mnist.load_data()


# In[3]:


train, test = data[0], data[1]


# In[4]:


train_examples, train_labels = train[0], train[1]
test_examples, test_labels = test


# In[5]:


rand_idx = random.randint(0, 60000-1)
example = train_examples[rand_idx]
label = train_labels[rand_idx]
img = Image.fromarray(example, mode='L')
print('Example #', rand_idx)
print('Label: ', label)
img.resize((252, 252))


# In[6]:


neural_network = tf.keras.Sequential([                  # SETTING UP LAYERS
    tf.keras.layers.Flatten(input_shape=(28,28)),       # input layer
    tf.keras.layers.Dense(370, activation='relu'),                         # hidden layer of 370 nodes
    tf.keras.layers.Dense(10, activation='softmax'),    # Output Layer --10 nodes for 10 possible outputs 
])


# In[7]:


neural_network.compile(
    optimizer='adam',                       # Choosing optimizer function
    loss='sparse_categorical_crossentropy', # Loss function is cross entropy -- Better than Mean Loss Square
    metrics=['accuracy']                    # Showing the accuracy of the machine
)


# In[8]:


neural_network.fit(train_examples, train_labels, epochs=10)# epochs makes the network go through the data set 10 times


# In[9]:


neural_network.summary()


# In[10]:


test_loss, test_accuracy = neural_network.evaluate(test_examples, test_labels, verbose=2)


# In[11]:


predictions = neural_network.predict(test_examples)


# In[12]:


predictions.shape


# In[15]:


predictions_argmax = np.argmax(predictions, axis=-1)


# In[65]:


rand_idx = random.randint(0, 10000-1)
example = test_examples[rand_idx]
label = test_labels[rand_idx]
prediction = predictions_argmax[rand_idx]
count = 0
while label == prediction:
    rand_idx = random.randint(0, 10000-1)
    example = test_examples[rand_idx]
    label = test_labels[rand_idx]
    prediction = predictions_argmax[rand_idx]
    count += 1
img = Image.fromarray(example, mode='L')
print('Label:', label)
print('Prediction:', prediction)
print('Amount of trials', count)
img.resize((252,252))


# In[46]:


cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(27, (3,3), padding='same', activation='relu', input_shape=(28,28,1)), # (3,3) is kernal size, 27 neurons, padding retains the shape of the shape-- dummy values around edge of image, input shape is 28 by 28 with 1 
    tf.keras.layers.Conv2D(31, (3,3), padding='same', activation='relu'), #larger kernal size for input, smaller for output
    tf.keras.layers.AvgPool2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[47]:


cnn.compile(
    optimizer='adam',                       # Choosing optimizer function
    loss='sparse_categorical_crossentropy', # Loss function is cross entropy -- Better than Mean Loss Square
    metrics=['accuracy']                    # Showing the accuracy of the machine
)


# In[48]:


cnn.summary()


# In[51]:


reshape_train = train_examples.reshape((60000, 28, 28, 1))


# In[52]:


reshape_train.shape


# In[53]:


cnn.fit(reshape_train, train_labels, epochs=10)

# In[ ]:


test_reshaped = test_examples.reshape((10000, 28, 28, 1))
tmp = test_reshaped
test_reshaped = tf.dtypes.cast(tmp, dtype=tf.float32)
test_loss, test_accuracy = cnn.evaluate(test_reshaped, test_labels, verbose=2)

predictions = cnn.predict(test_reshaped)
predictions_argmax = np.argmax(predictions, axis=-1)


# In[ ]:


count = 0
rand_idx = random.randint(0, 10000-1)
example = test_examples[rand_idx]
label = test_labels[rand_idx]
prediction = predictions_argmax[rand_idx]
img = Image.fromarray(example, mode='L')
print('Label:', label)
print('Prediction:', prediction)
print('Amount of trials', count)
img.resize((252,252))


export_path = "."
export_dir = os.path.dirname(export_path)
cnn.save("NeuralNetwork.h5")
