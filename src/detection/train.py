import argparse
import os

from ultralytics import YOLO, checks

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str,
                        default="yolov8n.pt", help='root path to '
                                                   'saved model or standard yolo model')
    parser.add_argument('--yaml', type=str, help='root path to yaml file')
    parser.add_argument('--epoch', type=int,
                        default=1, help='number of epoch')
    parser.add_argument('--imgsz', type=int,
                        default=128, help='image size')
    parser.add_argument('--batch', type=int,
                        default=1, help='number of sample in batch')
    parser.add_argument('--augment', type=bool,
                        default=False, help='enable data augmentation or not')

    return vars(parser.parse_args())


def train(model, yaml, epoch, imgsz, batch, augment):
    checks()
    model = YOLO(model)

    # Training
    model.train(
        data=yaml,
        imgsz=imgsz,
        epochs=epoch,
        batch=batch,
        pretrained=True,
        augment=True,
        name=f'lpd_{model}_{epoch}e_{batch}b_{"aug" if augment else ""}'
    )

    # Validation
    model.val(split='test')


if __name__ == '__main__':
    args = get_parser_args()
    train(
        model=args['model'],
        yaml=args['yaml'],
        epoch=args['epoch'],
        imgsz=args['imgsz'],
        batch=args['batch'],
        augment=args['augment'],
    )
