import argparse
import json
import os
from pathlib import Path

import cv2


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset_path', type=str,
                        default='recognize_dataset_root', help='root path to recognize dataset')
    return vars(parser.parse_args())


def check(path):
    lst = os.listdir(path)
    for k in lst:
        if ".txt" in k and k.replace(".txt", ".json") in lst:
            continue
        if ".json" in k and k.replace(".json", ".txt") in lst:
            continue
        print(k)


def main(path):
    ann_dir = Path(os.path.join(path, "ann"))
    for i, jsn in enumerate(list(map(str, list(ann_dir.glob("*.json"))))):
        with open(jsn) as js_file:
            data = json.load(js_file)
            label = data['description']

            with open(jsn.replace(".json", ".txt"), "w") as file:
                file.write(label)

        if i % 500 == 0:
            print(f"Data loaded: {i}")


def del_useless(path):
    lst = os.listdir(path)
    for i, k in enumerate(lst):
        if ".json" in k:
            os.remove(os.path.join(path, k))
        if i % 500 == 0:
            print(i)


def rename(path, start):
    lst = os.listdir(os.path.join(path, "ann"))
    for i, k in enumerate(lst):
        os.rename(os.path.join(path,
                               "ann",
                               k),
                  os.path.join(path,
                               "ann",
                               f"{start + i}.txt"))
        os.rename(os.path.join(path,
                               "img",
                               k.replace(".txt", ".png")),
                  os.path.join(path,
                               "img",
                               f"{start + i}.png"))


def color_inversion(path):
    lst = os.listdir(os.path.join(path, "img"))
    for k in lst:
        img = cv2.imread(os.path.join(path, "img", k))
        invert = cv2.bitwise_not(img)
        cv2.imwrite(os.path.join(path, "img", k), invert)


if __name__ == "__main__":
    args = get_parser_args()

    main(args['dataset_path'])
    del_useless(f"{args['dataset_path']}/ann")
    rename(args['dataset_path'], 0)
    color_inversion(args['dataset_path'])
