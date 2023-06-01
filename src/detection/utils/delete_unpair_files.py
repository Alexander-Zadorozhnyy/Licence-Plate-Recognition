import argparse
import os


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='root path to dir with yolo structure')

    return vars(parser.parse_args())


def delete_unpair_files(_path):
    paths = [os.path.join(_path, "train"),
             os.path.join(_path, "test"),
             os.path.join(_path, "valid"),
             ]

    for _path in paths:
        img_path = os.path.join(os.getcwd(), _path, "images")
        label_path = os.path.join(os.getcwd(), _path, "labels")

        images = os.listdir(img_path)
        img_txt = list(map(lambda x: ".".join(x.split(".")[:-1]) + ".txt", images))
        labels = os.listdir(label_path)

        singles = list(set(labels) - set(img_txt))
        singles_rev = list(set(img_txt) - set(labels))

        for single in singles:
            os.remove(os.path.join(label_path, single))

        for single in singles_rev:
            if single.replace('txt', "jpg") in images:
                os.remove(os.path.join(img_path, single.replace('txt', "jpg")))
            if single.replace('txt', "jpeg") in images:
                os.remove(os.path.join(img_path, single.replace('txt', "jpeg")))
            if single.replace('txt', "png") in images:
                os.remove(os.path.join(img_path, single.replace('txt', "png")))

        print(f'Folder {_path} done!')


if __name__ == '__main__':
    args = get_parser_args()
    delete_unpair_files(
        _path=args['path'],
    )
