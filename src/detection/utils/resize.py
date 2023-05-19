import argparse
import os

import cv2


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='root path to dir with yolo structure')
    parser.add_argument('--imgsz', type=int,
                        default=128, help='image size after resizing')

    return vars(parser.parse_args())


def resize_images(path, imgsz):
    paths = [os.path.join(path, "train"),
             os.path.join(path, "test"),
             os.path.join(path, "valid"),
             ]

    for path in paths:
        img_path = os.path.join(os.getcwd(), path, "images")
        images = os.listdir(img_path)

        for img in images:
            img_r = cv2.imread(os.path.join(img_path, img))
            img_r = cv2.resize(img_r, (imgsz, imgsz))

            cv2.imwrite(os.path.join(img_path, img), img_r)

        print(f'Folder {path} done!')


if __name__ == '__main__':
    args = get_parser_args()
    resize_images(
        path=args['path'],
        imgsz=args['imgsz'],
    )