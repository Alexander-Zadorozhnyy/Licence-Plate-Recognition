import argparse
import os

import cv2


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='root path to dir with yolo structure')
    parser.add_argument('--imgsz', type=int,
                        default=128, help='image size after resizing')

    return vars(parser.parse_args())


def resize_images(path_folder, imgsz):
    paths = [os.path.join(path_folder, "train"),
             os.path.join(path_folder, "test"),
             os.path.join(path_folder, "valid"),
             ]

    for _path_folder in paths:
        img_path = os.path.join(os.getcwd(), _path_folder, "images")
        images = os.listdir(img_path)

        for img in images:
            img_r = cv2.imread(os.path.join(img_path, img))
            img_r = cv2.resize(img_r, (imgsz, imgsz))

            cv2.imwrite(os.path.join(img_path, img), img_r)

        print(f'Folder {_path_folder} done!')


if __name__ == '__main__':
    args = get_parser_args()
    resize_images(
        path_folder=args['path'],
        imgsz=args['imgsz'],
    )
