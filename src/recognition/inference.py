import argparse
import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt
from mltu.utils.text_utils import get_cer, get_wer

from src.recognition.predict import ImageToWordModel


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', type=str,
                        default='model_path_root', help='root path to onnx model')
    parser.add_argument('--test_folder', type=str,
                        default='test_folder_root', help='root path to test folder')
    parser.add_argument('--mode', type=str,
                        default='all', help='all -- estimate wer and cer of all images, '
                                            'correct -- only correct images')
    parser.add_argument('--is_plot', type=bool,
                        default=True, help='plot examples or not')
    parser.add_argument('--coef_plot', type=int,
                        default=5, help='coef which used in plot function')
    return vars(parser.parse_args())


def plot_examples(model, test_folder):
    coef = 6
    _, axes = plt.subplots(coef, coef, figsize=(10, 5))

    for i, img in enumerate(os.listdir(test_folder)[:coef ** 2]):
        image_path = f"{test_folder}/{img}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prediction_text = model.predict(image)

        img = (image * 255).astype(float)
        axes[i // coef, i % coef].imshow(img[:, :], cmap="gray")
        axes[i // coef, i % coef].set_title(prediction_text)
        axes[i // coef, i % coef].axis("off")

    plt.show()


def correct_inference(model, test_folder):
    all_c = len(os.listdir(test_folder))
    count = 0

    for img in os.listdir(test_folder):
        image_path = f"{test_folder}/{img}"
        label = img.split(".")[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        prediction_text = model.predict(image)

        wer = get_wer(prediction_text, label)
        count = count + 1 if wer == 0 else count

    print(f"Inference right {count} of all {all_c}. Therefore right {count / all_c * 100}%")


def inference(model, test_folder):
    all_c = len(os.listdir(test_folder))
    count = 0
    accum_cer, accum_wer = [], []

    for img in os.listdir(test_folder):

        image_path = f"{test_folder}/{img}"
        label = img.split(".")[0]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        start = time.time()
        prediction_text = model.predict(image)
        end = time.time()
        print("The time of execution of above program is :",
              (end - start) * 10 ** 3, "ms")

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)

        if wer == 0:
            count += 1
            print("Image: ", image_path)
            print("Label:", label)
            print("Prediction: ", prediction_text)
            print(f"CER: {cer}; WER: {wer}")

            cv2.imshow(prediction_text, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        accum_cer.append(cer)
        accum_wer.append(wer)
        print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
    print(f"Inference right {count} of all {all_c}. Therefore right {count / all_c * 100}%")


if __name__ == "__main__":
    args = get_parser_args()

    model = ImageToWordModel(
        model_path=args['model_path'],
    )
    if args['is_plot']:
        plot_examples(model, args['test_folder'])

    if args['mode'] == 'all':
        inference(model, args['test_folder'])
    elif args['mode'] == 'correct':
        correct_inference(model, args['test_folder'])
    else:
        print("Choose 'all' or 'correct' mode")

    # "E:\MachineLearningProjects\LicencePlateRecognition_ResearchProject\
    # Licence-Plate-Recognition\src\pretrained_models\MRNET_100ep_512b-05_17_21_40"
    # "E:/MachineLearningProjects/LicencePlateRecognition_ResearchProject/
    # datasets/ocr_txt/test/img"
