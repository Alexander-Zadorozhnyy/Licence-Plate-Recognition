import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *


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
            cvzone.putTextRect(img, f'licence_plate-{conf}', (max(0, x1), max(35, y1 - 5)),
                               scale=1.5, thickness=2, offset=2)

    return img


def get_all_detections(results, height, width):
    coef_h, coef_w = height / 512, width / 512
    classNames = ["licence_plate", "vehicle"]

    detections = {x: [] for x in classNames}

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1 * coef_w), int(y1 * coef_h), int(x2 * coef_w), int(y2 * coef_h)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            # if currentClass == "vehicle" and conf > 0.2:
            #     continue
            detections[currentClass].append([x1, y1, x2, y2, conf])

    return detections


def highlight_all_detections(detections, img):
    # print(detections)
    for key, val in detections.items():
        for obj in val:
            x1, y1, x2, y2, conf = obj
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(obj)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img,
                              (x1, y1, w, h),
                              l=9,
                              rt=2,
                              colorR=(255, 0, 255) if key == "vehicle" else (255, 0, 0),
                              colorC=(0, 255, 0) if key == "vehicle" else (0, 0, 128))
            cvzone.putTextRect(img,
                               f'{key}-{conf}',
                               (max(0, x1), max(35, y1 - 5)),
                               scale=1.5,
                               thickness=2,
                               offset=2,
                               colorR=(255, 0, 255) if key == "vehicle" else (255, 0, 0))

    return img


def save_vid(source, weights):
    model = YOLO(f"../weights/{weights}")
    model.predict(source, save=True, imgsz=512, conf=0.1, visualize=True)


def main(source, weights, size=512, mode="all"):
    cap = cv2.VideoCapture(f"../data/videos/{source}")  # For Video
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))
    model = YOLO(f"../weights/{weights}", task="detect")

    classNames = ["licence_plate", 'vehicle']  # , "vehicle"

    # Tracking
    # tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.1)

    # totalCount = []

    while True:
        success, img = cap.read()
        if not success:
            break
        img_copy = img.copy()
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = cv2.Canny(image=img, threshold1=100, threshold2=200)
        height, width, channels = img.shape
        img = cv2.resize(img, dsize=(size, size))
        print(img.shape)
        results = model(img, stream=True, task="detect", imgsz=(512, 512))
        # coef_h, coef_w = height / 512, width / 512

        detections = get_all_detections(results, height, width)
        # detections = {key: [] for key in ['licence_plate', 'vehicle']}  # np.empty((0, 5))
        #
        # for r in results:
        #     boxes = r.boxes
        #     for box in boxes:
        #         # Bounding Box
        #         x1, y1, x2, y2 = box.xyxy[0]
        #         x1, y1, x2, y2 = int(x1 * coef_w), int(y1 * coef_h), int(x2 * coef_w), int(y2 * coef_h)
        #         # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
        #         w, h = x2 - x1, y2 - y1
        #
        #         # Confidence
        #         conf = float(box.conf[0])  # math.ceil((box.conf[0] * 100)) / 100
        #         # Class Name
        #         cls = int(box.cls[0])
        #         currentClass = classNames[cls]
        #
        #         if (currentClass == "licence_plate" or currentClass == "vehicle") and conf > 0:
        #             # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
        #             #                    scale=0.6, thickness=1, offset=3)
        #             # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
        #             print(conf)
        #             currentArray = np.array([x1, y1, x2, y2, conf])
        #             detections[currentClass].append(currentArray)

        img_p = None

        if mode == "plate":
            img_p = highlight_licence_plate(detections, img_copy)
        if mode == "all":
            img_p = highlight_all_detections(detections, img_copy)

        # img = cv2.resize(img, dsize=(width, height))
        #
        # resultsTracker = tracker.update(detections)

        # for result in resultsTracker:
        #     x1, y1, x2, y2, conf = result
        #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        #     print(result)
        #     w, h = x2 - x1, y2 - y1
        #     cvzone.cornerRect(img_copy, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        #     cvzone.putTextRect(img_copy, f' {int(conf)}', (max(0, x1), max(35, y1)),
        #                        scale=2, thickness=3, offset=10)

        # cx, cy = x1 + w // 2, y1 + h // 2
        # cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        #
        # if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
        #     if totalCount.count(id) == 0:
        #         totalCount.append(id)
        #         cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # if totalCount.count(id) == 0:
        #     totalCount.append(id)
        out.write(img_p)
        cv2.imshow("Image", img_p)
        # cv2.imshow("ImageRegion", imgRegion)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # save_vid(
    #     source="E:/MachineLearningProjects/LicencePlateRecognition_ResearchProject/yolov8/data/videos/real_example.mp4",
    #     weights="best_512_59ep_64b_aug_s.pt")
    main(source="car_show.mp4",
         weights="best_512_109ep_64b_aug_s.pt",
         mode="all")
