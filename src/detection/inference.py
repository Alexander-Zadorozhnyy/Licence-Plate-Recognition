import argparse
import time

import cv2
import tensorflow as tf
import numpy as np


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str,
                        default='model_path_root', help='root path to model')
    parser.add_argument('--img_path', type=str,
                        default='img_path', help='path to image')
    return vars(parser.parse_args())


def main(model_path, img_path):
    # Load the TFLite model and allocate tensors. View details
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on input data.
    input_shape = input_details[0]['shape']
    print(input_shape)

    # Use same image as Keras model
    img = cv2.imread(img_path)
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
    print(output_data)

    end = time.time()
    print("The time of execution of above program is :",
          (end - start) * 10 ** 3, "ms")


if __name__ == "__main__":
    args = get_parser_args()
    main(model_path=args['model_path'], img_path=args['img_path'])
