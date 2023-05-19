import argparse
import os


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='root path to dir with yolo structure')

    return vars(parser.parse_args())


def delete_unpair_files(path):
    paths = [os.path.join(path, "train"),
             os.path.join(path, "test"),
             os.path.join(path, "valid"),
             ]

    for path in paths:
        img_path = os.path.join(os.getcwd(), path, "images")
        label_path = os.path.join(os.getcwd(), path, "labels")

        images = os.listdir(img_path)
        img_txt = list(map(lambda x: ".".join(x.split(".")[:-1]) + ".txt", images))
        labels = os.listdir(label_path)

        singles = list(set(labels) - set(img_txt))
        singles_rev = list(set(img_txt) - set(labels))

        for s in singles:
            os.remove(os.path.join(label_path, s))

        for s in singles_rev:
            if s.replace('txt', "jpg") in images:
                os.remove(os.path.join(img_path, s.replace('txt', "jpg")))
            if s.replace('txt', "jpeg") in images:
                os.remove(os.path.join(img_path, s.replace('txt', "jpeg")))
            if s.replace('txt', "png") in images:
                os.remove(os.path.join(img_path, s.replace('txt', "png")))

        print(f'Folder {path} done!')


if __name__ == '__main__':
    args = get_parser_args()
    delete_unpair_files(
        path=args['path'],
    )
