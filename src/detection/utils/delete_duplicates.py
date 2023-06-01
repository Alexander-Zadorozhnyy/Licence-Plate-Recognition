import argparse
import os

from imagededup.methods import DHash


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type=str, help='root path to dir with yolo structure')
    parser.add_argument('--max_distance_threshold', type=int,
                        default=1, help='maxi distance of threshold')

    return vars(parser.parse_args())


def delete_duplicates(path, max_threshold):
    dirs = ['train', 'test', 'valid']
    for _dir in dirs:
        path = os.path.join(path, _dir, "images")
        method_object = DHash()
        duplicates = method_object.find_duplicates_to_remove(image_dir=path,
                                                             max_distance_threshold=max_threshold)
        for i in duplicates:
            os.remove(os.path.join(path, i))


if __name__ == "__main__":
    args = get_parser_args()
    delete_duplicates(
        path=args['path'],
        max_threshold=args['max_distance_threshold'],
    )
