import math

import cv2
import cvzone


def is_in_interval(point, start, end):
    return start <= point <= end


def is_plate(plate, vehicles):
    x_1, y_1, x_2, y_2, _ = plate
    for vehicle in vehicles:
        v_x1, v_y1, v_x2, v_y2, _ = vehicle
        if is_in_interval(x_1, v_x1, v_x2) \
                and is_in_interval(x_2, v_x1, v_x2) \
                and is_in_interval(y_1, v_y1, v_y2) \
                and is_in_interval(y_2, v_y1, v_y2):
            return True
    return False


def highlight_licence_plate(detections, img, model_plate):
    plates = []
    for plate in detections['licence_plate']:
        if is_plate(plate, detections['vehicle']):
            plates += [plate]
            x_1, y_1, x_2, y_2, _ = plate
            width, height = x_2 - x_1, y_2 - y_1
            img_r = img[y_1:y_2, x_1:x_2]
            img_r = cv2.cvtColor(img_r, cv2.COLOR_RGB2GRAY)
            prediction_text = model_plate.predict(img_r)
            cvzone.cornerRect(img, (x_1, y_1, width, height), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'{prediction_text}', (max(0, x_1), max(35, y_1 - 5)),
                               scale=1.5, thickness=2, offset=2)

    return plates, img


def get_all_detections(results, height, width):
    coef_h, coef_w = height / 512, width / 512
    class_names = ["licence_plate", "vehicle"]

    detections = {x: [] for x in class_names}

    for res in results:
        boxes = res.boxes
        for box in boxes:
            # Bounding Box
            x_1, y_1, x_2, y_2 = box.xyxy[0]
            x_1, y_1, x_2, y_2 = int(x_1 * coef_w), int(y_1 * coef_h), \
                int(x_2 * coef_w), int(y_2 * coef_h)
            # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            current_class = class_names[cls]
            # if currentClass == "vehicle" and conf > 0.2:
            #     continue
            detections[current_class].append([x_1, y_1, x_2, y_2, conf])

    return detections


def highlight_all_detections(detections, img):
    # print(detections)
    for key, val in detections.items():
        for obj in val:
            x_1, y_1, x_2, y_2, conf = obj
            # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # print(obj)
            width, height = x_2 - x_1, y_2 - y_1
            cvzone.cornerRect(img,
                              (x_1, y_1, width, height),
                              l=9,
                              rt=2,
                              colorR=(255, 0, 255) if key == "vehicle" else (255, 0, 0),
                              colorC=(0, 255, 0) if key == "vehicle" else (0, 0, 128))
            cvzone.putTextRect(img,
                               f'{key}-{conf}',
                               (max(0, x_1), max(35, y_1 - 5)),
                               scale=1.5,
                               thickness=2,
                               offset=2,
                               colorR=(255, 0, 255) if key == "vehicle" else (255, 0, 0))

    return img
