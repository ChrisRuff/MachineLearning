from PIL import Image
from io import BytesIO
import cv2
import base64
import tensorflow as tf
import numpy as np


myModel = tf.keras.models.load_model("simplenetowrk.h5")


def convert(base64Data):
    base64Data = base64Data[22:]
    im = Image.open(BytesIO(base64.b64decode(base64Data)))
    im.save("./image.png")
    return predict("./image.png")

def predict(file_dir):
    data = cv2.imread(file_dir, cv2.IMREAD_UNCHANGED)
    data = data[:, :, 3]
    resized = cv2.resize(data, dsize=(28, 28))
    test_data = np.asarray(resized)
    cv2.imshow('ImageWindow', test_data)
    cv2.waitKey(0)
    test_data = test_data.reshape((1, 28, 28))   
    # test_data = np.expand_dims(test_data, axis=0)
    # test_data = tf.image.decode_png(reshaped, dtype=tf.dtypes.float32)
    test_data = tf.dtypes.cast(test_data, dtype=tf.float32) 
    for x in test_data:
        print(x)
    predictions = myModel.predict(test_data)
    print(predictions)
    predictions_argmax = np.argmax(predictions, axis=-1)
    return predictions_argmax


