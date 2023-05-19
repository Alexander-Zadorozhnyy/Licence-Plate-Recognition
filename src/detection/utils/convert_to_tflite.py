import argparse

from ultralytics import YOLO, checks


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str,
                        default="yolov8n.pt", help='root path to saved_model')
    parser.add_argument('--imgsz', type=int,
                        default=128, help='image size')
    parser.add_argument('--quantization', type=bool,
                        default=False, help='enable quantization or not')

    return vars(parser.parse_args())


def convert_to_tflite(model, imgsz, quantization):
    checks()
    model = YOLO(model)

    # Export the model to tflite format
    model.export(format='tflite', imgsz=imgsz, int8=quantization, half=quantization)


if __name__ == '__main__':
    args = get_parser_args()
    convert_to_tflite(
        model=args['model'],
        imgsz=args['imgsz'],
        quantization=args['quantization']
    )
