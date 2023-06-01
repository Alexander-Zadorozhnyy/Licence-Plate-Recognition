import math

import cv2
import cvzone


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


def highlight_licence_plate(detections, img, model_plate):
    plates = []
    for plate in detections['licence_plate']:
        if is_plate(plate, detections['vehicle']):
            plates += [plate]
            x1, y1, x2, y2, conf = plate
            w, h = x2 - x1, y2 - y1
            img_r = img[y1:y2, x1:x2]
            img_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
            prediction_text = model_plate.predict(img_r)
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'{prediction_text}', (max(0, x1), max(35, y1 - 5)),
                               scale=1.5, thickness=2, offset=2)

    return plates, img


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