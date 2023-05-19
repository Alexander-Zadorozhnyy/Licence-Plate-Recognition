import argparse
import math
import os

from ultralytics import YOLO

CLASSNAMES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='root path to dir with yolo structure')
    parser.add_argument('--conf_treshold', type=float,
                        default=0.5, help='confidence treshold used in yolov8x model')

    return vars(parser.parse_args())


def label_vehicles(model, path, conf_treshold):
    paths = [os.path.join(path, "train"),
             os.path.join(path, "test"),
             os.path.join(path, "valid"),
             ]

    for path in paths:
        img_path = os.path.join(os.getcwd(), path, "images")
        label_path = os.path.join(os.getcwd(), path, "labels")
        images = os.listdir(img_path)

        for img in images:
            detections = []

            # Use the model
            results = model(os.path.join(img_path, img), stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding Box
                    coords = list(map(float, box.xywhn[0]))

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    currentClass = CLASSNAMES[cls]

                    if (currentClass == "car" or currentClass == "truck"
                        or currentClass == "bus" or currentClass == "motorbike") \
                            and conf > conf_treshold:
                        detections.append(coords)

            with open(os.path.join(label_path, f"{'.'.join(img.split('.')[:-1])}.txt"), mode='r+') as f:
                file = f.read()
                for d in detections:
                    if file != "":
                        f.write(f'\n1 {" ".join(["%.3f" % i for i in d])}')
                    else:
                        f.write(f'1 {" ".join(["%.3f" % i for i in d])}')

        print(f'Folder {path} done!')


if __name__ == '__main__':
    # Load a model
    model = YOLO("weights/yolov8x.pt")

    label_vehicles(model, "512_all")

    args = get_parser_args()
    label_vehicles(
        model=model,
        path=args['path'],
        conf_treshold=args['conf_treshold'],
    )
