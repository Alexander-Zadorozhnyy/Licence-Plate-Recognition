import argparse
import math
import os

from ultralytics import YOLO

CLASSNAMES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
              "truck", "boat", "traffic light", "fire hydrant", "stop sign",
              "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
              "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove",
              "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
              "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
              "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
              "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
              "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
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

    for _path in paths:
        img_path = os.path.join(os.getcwd(), _path, "images")
        label_path = os.path.join(os.getcwd(), _path, "labels")
        images = os.listdir(img_path)

        for img in images:
            detections = []

            # Use the model
            results = model(os.path.join(img_path, img), stream=True)

            for res in results:
                boxes = res.boxes
                for box in boxes:
                    # Bounding Box
                    coords = list(map(float, box.xywhn[0]))

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Class Name
                    cls = int(box.cls[0])
                    current_class = CLASSNAMES[cls]

                    if current_class in ("car", "truck", "bus", "motorbike") \
                            and conf > conf_treshold:
                        detections.append(coords)

            with open(os.path.join(label_path, f"{'.'.join(img.split('.')[:-1])}.txt"),
                      mode='r+') as file_orig:
                file = file_orig.read()
                for detect in detections:
                    if file != "":
                        file_orig.write(f"\n1 {' '.join(['%.3f' % i for i in detect])}")
                    else:
                        file_orig.write(f"1 {' '.join(['%.3f' % i for i in detect])}")

        print(f'Folder {_path} done!')


if __name__ == '__main__':
    # Load a model
    model = YOLO("weights/yolov8x.pt")

    label_vehicles(model, "512_all", 0.3)

    args = get_parser_args()
    label_vehicles(
        model=model,
        path=args['path'],
        conf_treshold=args['conf_treshold'],
    )
