import argparse

from ultralytics import YOLO

from detection.image import modified_image_plate_recognition
from detection.video import modified_video_plate_recognition
from src.recognition.predict import ImageToWordModel


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--source', type=str,
                        default="../data/images/example.jpg",
                        help='root path to input file')
    parser.add_argument('--detection_model', type=str,
                        default="../src/pretrained_models/YOLOv8s_detection.pt",
                        help='root path to saved detection model or standard yolo model')
    parser.add_argument('--recognition_model', type=str,
                        default="../src/pretrained_models/MRNet_recognition",
                        help='root path to saved recognition model or standard yolo model')
    parser.add_argument('--size', type=int,
                        default=512, help='image transform size')

    parser.add_argument('--plate', type=bool,
                        default=False, help='detect vehicles and plates or only plates')

    return vars(parser.parse_args())


def main(source, detection_model, recognition_model, size, mode):
    detection = YOLO(detection_model,
                     task="detect")

    recognition = ImageToWordModel(
        model_path=recognition_model,
    )

    if ".mp4" in source:
        modified_video_plate_recognition(source=source,
                                         detection_model=detection,
                                         recognition_model=recognition,
                                         size=size,
                                         mode=mode)
    if any(x in source for x in ['.png', '.jpeg', '.jpg']):
        modified_image_plate_recognition(source=source,
                                         detection_model=detection,
                                         recognition_model=recognition,
                                         size=size,
                                         mode=mode)
    else:
        print("Incorrect format of input file")


if __name__ == "__main__":
    args = get_parser_args()
    main(source=args['source'],
         detection_model=args['detection_model'],
         recognition_model=args['recognition_model'],
         size=args['size'],
         mode='plate' if args['plate'] else 'all')
