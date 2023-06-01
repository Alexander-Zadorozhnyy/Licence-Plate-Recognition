![logo](https://i.ibb.co/LdFB56X/lpr-logo.png)

![version](https://img.shields.io/badge/Version-Alpha--0.0.1-blue)
![issues](https://img.shields.io/github/issues/Alexander-Zadorozhnyy/Licence-Plate-Recognition)
![forks](https://img.shields.io/github/forks/Alexander-Zadorozhnyy/Licence-Plate-Recognition)
![stars](https://img.shields.io/github/stars/Alexander-Zadorozhnyy/Licence-Plate-Recognition)
![license](https://img.shields.io/github/license/Alexander-Zadorozhnyy/Licence-Plate-Recognition)

# Description

🛠 
Introducing an innovative license plate detection and recognition tool that breaks through the barriers of computing power limitations. This lightweight solution leverages the power of YOLOv8s and MRNetm models to achieve real-time license plate recognition on devices with constrained resources. YOLOv8s, a state-of-the-art object detection model, efficiently detects license plates with remarkable accuracy. MRNetm, a compact deep learning model, specializes in recognizing and extracting alphanumeric characters from license plates.

This tool has a wide range of applications, including traffic enforcement, parking management, and vehicle tracking. It enables automated toll collection, traffic violation detection, and efficient vehicle identification in real time. With its ability to operate on low-powered devices such as smartphones, surveillance cameras, and edge devices, the tool offers scalability and versatility.

By harnessing YOLOv8s and MRNetm, this license plate detection and recognition tool overcomes the limitations of computing power, making it a valuable solution for traffic management systems, parking facilities, and law enforcement agencies. Its lightweight nature and real-time capabilities pave the way for enhanced efficiency, security, and automation in various domains.

The work was carried out with the support and on the basis of the laboratory of theoretical and interdisciplinary problems of Informatics of the Federal State Budgetary Institution of Science "St. Petersburg Federal Research Center of the Russian Academy of Sciences" (St. Petersburg FRC RAS). Official website: https://dscs.pro/

## Licence Plate Recognition Example

<p align="center">
      <img src="https://i.ibb.co/QcLXCmf/res.png" alt="Captcha examples that can be solved by this CAPTCHA solver" width="726">
</p>

## Usage guide⚙️
##### Step0: Clone the Project
```shell
git clone https://github.com/Alexander-Zadorozhnyy/Licence-Plate-Recognition.git
cd Licence-Plate-Recognition
```
##### Step1: Create & Activate Conda Env (Optional)
```shell
conda create -n "Licence-Plate-Recognition" python=3.10
conda activate Licence-Plate-Recognition
```
##### Step2: Install PIP Requirements 
```shell
pip install -r requirement.txt
```
##### Step3: Run app/main.py
If you want only to test app, all params have default values
```shell
python -m app.main.py --source --detection_model --recognition_model --size --plate
```

## Training guide ‍🔬
##### Step0: Clone the Project
```shell
git clone https://github.com/Alexander-Zadorozhnyy/Licence-Plate-Recognition.git
cd Licence-Plate-Recognition
```
##### Step1: Create & Activate Conda Env (Optional)
```shell
conda create -n "Licence-Plate-Recognition" python=3.10
conda activate Licence-Plate-Recognition
```
##### Step2: Install PIP Requirements 
```shell
pip install -r requirement.txt
```
##### Step3: Configure src/detection/yaml/your_file.yaml

##### Step4: Add your dataset to src/datasets folder. To create clear and efficient dataset you canuse src/detection/utils functions

##### Step5: Train YOLO model
```shell
python -m src.detection.train --model model_name --yaml path_to_yaml --epoch number_of_epoch --imgsz size_of_images --batch bathc_size --augment True
```

##### Step6: Configure src/recognition/config.py

##### Step7: Train MRNet model
```shell
python -m src.recognition.train.py --train_path path_to_train_folder --valid_path path_to_valid_folder --augment True --saved_model_name save_path --save_csv True
 ```

## Metrics

Licence Plate Detector

    Time: 23.4ms for 1 img

    Accuracy: ~96%

Licence Plate Recognizer

    Time: 25ms for 1 img

    Accuracy:  ~90%

## Documentation

> You can check some details about this solver in the [docs](https://github.com/Alexander-Zadorozhnyy/Licence-Plate-Recognition/tree/main/docs) directory:
> - docs/report.pdf - educational practice's report (RU)

## Authors

- [@ZadorozhnyyA](https://github.com/Alexander-Zadorozhnyy)

## License

Source code of this repository is released under 
the [Apache-2.0 license](https://choosealicense.com/licenses/apache-2.0/)