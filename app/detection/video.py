from ultralytics import YOLO
import cv2

from app.detection.utils import get_all_detections, \
    highlight_licence_plate, highlight_all_detections
from src.recognition.predict import ImageToWordModel


def default_video_plate_detection(source, model):
    model.predict(source, save=True, imgsz=512, conf=0.1, visualize=True)


def modified_video_plate_recognition(source, detection_model,
                                     recognition_model, size=512, mode="all"):
    cap = cv2.VideoCapture(source)  # For Video
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

    while True:
        success, img = cap.read()
        if not success:
            break

        img_copy = img.copy()
        img_p = None

        height, width, _ = img.shape
        img = cv2.resize(img, dsize=(size, size))

        results = detection_model(img, stream=True, task="detect", imgsz=(512, 512))
        detections = get_all_detections(results, height, width)

        if mode == "plate":
            _, img_p = highlight_licence_plate(detections, img_copy, recognition_model)
        if mode == "all":
            img_p = highlight_all_detections(detections, img_copy)

        out.write(img_p)
        cv2.imshow("Image", img_p)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detection = YOLO("../../src/pretrained_models/YOLOv8s_detection.pt",
                     task="detect")

    recognition = ImageToWordModel(
        model_path="../../src/pretrained_models/MRNet_recognition",
    )

    # default_video_plate_detection(
    #     source="E:/MachineLearningProjects/
    #     LicencePlateRecognition_ResearchProject/yolov8/data/videos/real_example.mp4",
    #     model=detection)

    modified_video_plate_recognition(source="../data/videos/example.mp4",
                                     detection_model=detection,
                                     recognition_model=recognition,
                                     mode="plate")
