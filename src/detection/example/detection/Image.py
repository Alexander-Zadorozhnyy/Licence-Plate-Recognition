import os

from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import torch
import numpy as np
import torchvision as tv

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.transformers import ImageResizer

from src.recognition.config import SYMBOLS


class ImageToWordModel(OnnxInferenceModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.char_list = sorted([i for i in SYMBOLS])

    def predict(self, image: np.ndarray):
        image = ImageResizer.resize_maintaining_aspect_ratio(image, *self.input_shape[:2][::-1])
        image_pred = np.expand_dims(image, axis=0).astype(np.float32)
        image_pred = np.expand_dims(image_pred, axis=-1).astype(np.float32)
        # print(image_pred.shape)
        preds = self.model.run(None, {self.input_name: image_pred})[0]

        return ctc_decoder(preds, self.char_list)[0]


def get_all_detections(results):
    classNames = ["licence_plate", "vehicle"]

    detections = {x: [] for x in classNames}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            detections[currentClass].append([x1, y1, x2, y2, conf])

    return detections


def is_in_interval(point, start, end):
    return start <= point <= end


def is_plate(plate, vehicles):
    x1, y1, x2, y2, conf = plate
    for vehicle in vehicles:
        v_x1, v_y1, v_x2, v_y2, conf = vehicle
        if is_in_interval(x1, v_x1, v_x2) \
                and is_in_interval(x2, v_x1, v_x2) \
                and is_in_interval(y1, v_y1, v_y2) \
                and is_in_interval(y2, v_y1, v_y2):
            return True
    return False


def highlight_licence_plate(detections, img):
    plates = []
    for plate in detections['licence_plate']:
        if is_plate(plate, detections['vehicle']):
            plates += [plate]
            x1, y1, x2, y2, conf = plate
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, 'licence_plate', (max(0, x1), max(35, y1 - 5)),
                               scale=1.5, thickness=2, offset=2)

    return plates, img


def highlight_all_detections(detections, img):
    # print(detections)
    for key, val in detections.items():
        for obj in val:
            x1, y1, x2, y2, conf = obj
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(obj)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'{key}', (max(0, x1), max(35, y1 - 5)),
                               scale=1.5, thickness=2, offset=2)

    return img


def crop_plates(plates, img):

    results = []
    for plate in plates:
        x1, y1, x2, y2, conf = plate
        # w, h = x2 - x1, y2 - y1
        img_r = img[y1:y2, x1:x2]
        # print(img_r.shape)
        img_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
        results += [img_r]
    return results



def test_speed(model, image):
    start = time.time()
    path = "../../datasets/512_all_right/512_all/test/images"
    for img in os.listdir(path):
        img = cv2.imread(os.path.join(path, img))
        img = cv2.resize(img, dsize=(512, 512))

        results = model(img, stream=True, )
        detections = get_all_detections(results)
        plates, highlighted_img = highlight_licence_plate(detections, img)

    end = time.time()
    print("The time of execution of above program is :",
          (end - start) * 10 ** 3, "ms")
    # cv2.imshow("Image", highlighted_img)
    # cv2.waitKey(0)


def main(model, image, mode, model_plate=None):
    img = cv2.imread(image)
    img = cv2.resize(img, dsize=(512, 512))

    highlighted_img = None

    start = time.time()
    results = model.predict(img, imgsz=512, device='cpu')
    detections = get_all_detections(results)

    if mode == 'all':
        highlighted_img = highlight_all_detections(detections, img)

    if mode == 'plate':
        plates, highlighted_img = highlight_licence_plate(detections, img.copy())
        plates = crop_plates(plates, img)
        for plate in plates:
            prediction_text = model_plate.predict(plate)
            print(prediction_text)

    end = time.time()
    print("The time of execution of above program is :",
          (end - start) * 10 ** 3, "ms")

    print(end - start)
    cv2.imshow("Image", highlighted_img)
    cv2.waitKey(0)


def find_contours(img, model, conf_threshold=0.75, iou_threshold=0.1, width=256, heigth=256):
    frame = img.copy()
    # img = cv2.resize(img, (width, heigth))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(img).unsqueeze(0).transpose(1, 3) / 255.0
    preds = model(tensor)[0]
    # print(preds)
    preds = preds[preds[:, 4] > conf_threshold]

    center_x, center_y = preds[:, 0], preds[:, 1]
    box_w, box_h = preds[:, 2], preds[:, 3]
    scores = preds[:, 4]
    preds = torch.stack([
        torch.floor(center_x - box_w / 2),
        torch.floor(center_y - box_h / 2),
        torch.ceil(center_x + box_w / 2),
        torch.ceil(center_y + box_h / 2)
    ], dim=1)

    indexes = tv.ops.nms(preds, scores, iou_threshold)
    preds = preds[indexes].cpu().numpy().astype(np.int32).tolist()

    boxes = []
    for pred in preds:
        x1, y1, x2, y2 = pred
        nx1, nx2 = int(x1 / width * frame.shape[1]), int(x2 / width * frame.shape[1])
        ny1, ny2 = int(y1 / heigth * frame.shape[0]), int(y2 / heigth * frame.shape[0])
        frame = cv2.rectangle(frame, (ny2, nx1), (ny1, nx2), (0, 0, 255), 1)
        boxes.append((x1 / width, y1 / heigth, x2 / width, y2 / heigth))

    return frame, boxes


def test_v5_speed(model, image):
    # start = time.time()
    # img = cv2.imread(image)
    # # img = cv2.resize(img, dsize=(640, 640))
    # frames = find_contours(img, model)[0]
    #
    # end = time.time()
    # print("The time of execution of above program is :",
    #       (end - start) * 10 ** 3, "ms")

    # start = time.perf_counter()
    # img = cv2.imread(image)
    # # img = cv2.resize(img, dsize=(640, 640))
    # frames = find_contours(img, model)[0]
    #
    # end = time.perf_counter()
    # print(end - start)
    path = "../../datasets/few_licence_plate_numeric/train/images"
    start = time.time()
    for img in os.listdir(path):
        img = cv2.imread(os.path.join(path, img))
        img = cv2.resize(img, dsize=(512, 512))
        frames = find_contours(img, model)[0]

    end = time.time()
    print("The time of execution of above program is :",
          (end - start) * 10 ** 3, "ms")
    # cv2.imshow("Image", frames)
    # cv2.waitKey(0)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # model = torch.hub.load('ultralytics/yolov5', 'custom', path='../weights/prev_yolov5.pt')
    # model = model.to(device)
    # test_v5_speed(model=model, image="../data/images/police.jpg")
    print("===========================")
    model = YOLO("E:/MachineLearningProjects/LicencePlateRecognition_ResearchProject/Licence-Plate-Recognition/src/detection/weights/best_512_109ep_64b_aug_s.pt")
    # model.to('cuda')
    model_plate = ImageToWordModel(
        model_path="E:\MachineLearningProjects\LicencePlateRecognition_ResearchProject\Licence-Plate-Recognition\src\pretrained_models\MRNET_100ep_512b-05_17_21_40",
    )
    # test_speed(model, "../data/images/123.jpeg")
    # test_v5_speed(model, "../data/images/123.jpeg")
    main(model=model,
         model_plate=model_plate,
         image="../data/images/rus.jpg",
         mode='plate')  # plate or all

    # main(model=model,
    #      image="../data/images/car4.jpg",
    #      mode='plate')  # plate or all

    # model = model.to(device)
    # test_speed(model=model,
    #      image="../Images/car5.jpg")
    # test_v5_speed(model=model, image="../data/images/test.jpg")
    #
    # test_speed(model="../prev_yolov5.pt",
    #            image="../Images/car5.jpg")
