import numpy as np
import tensorflow as tf
import cv2
from skimage import color
from skimage import io

imgGray = io.imread('download.jpeg')
img = color.rgb2gray(imgGray)

img = cv2.resize(img, (28,28))
img = np.array(img, dtype="float32")
img = np.reshape(img, (1,28,28,1))

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input.shape = input_details[0]['shape']

interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)