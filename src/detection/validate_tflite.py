import time

import cv2
import tensorflow as tf
import numpy as np

# Load the TFLite model and allocate tensors. View details
interpreter = tf.lite.Interpreter(model_path="weights/model_float16.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data.
input_shape = input_details[0]['shape']
print(input_shape)

# Use same image as Keras model
img = cv2.imread("data/images/test.jpg")
img = cv2.resize(img, dsize=(512, 512))

start = time.time()
input_data = np.array(img)
input_data = np.expand_dims(input_data, axis=0)
input_data = np.array(input_data, dtype=np.float32)
print(input_data.shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
# print(output_data)

end = time.time()
print("The time of execution of above program is :",
          (end - start) * 10 ** 3, "ms")