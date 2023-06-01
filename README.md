![logo](https://i.ibb.co/YQ4qCPm/Color-logo-with-background.png)

![version](https://img.shields.io/badge/Version-Alpha--0.0.1-blue)
![issues](https://img.shields.io/github/issues/Alexander-Zadorozhnyy/Licence-Plate-Recognition)
![forks](https://img.shields.io/github/forks/Alexander-Zadorozhnyy/Licence-Plate-Recognition)
![stars](https://img.shields.io/github/stars/Alexander-Zadorozhnyy/Licence-Plate-Recognition)
![license](https://img.shields.io/github/license/Alexander-Zadorozhnyy/Licence-Plate-Recognition)

# Description

🛠 Licence plate detection and recognition tool. A new lightweight tool is able to recognize license plates in real time on devices with significant limitations in computing power.

The work was carried out with the support and on the basis of the laboratory of theoretical and interdisciplinary problems of Informatics of the Federal State Budgetary Institution of Science "St. Petersburg Federal Research Center of the Russian Academy of Sciences" (St. Petersburg FRC RAS). Official website: https://dscs.pro/

## CAPTCHA image recognition

There are CAPTCHA that can be recognized by this solver. You can find some trained models in app/models.
<p align="center">
      <img src="https://i.ibb.co/mGM2wRx/cap.png" alt="Captcha examples that can be solved by this CAPTCHA solver" width="726">
</p>

## Requirements

```shell
pip install -r requirement.txt
```

## Usage guide⚙️
##### Step0: Clone the Project
```shell
git clone https://github.com/Alexander-Zadorozhnyy/CGAN_CAPTCHA_SOLVER.git
cd CGAN_CAPTCHA_SOLVER
```
##### Step1: Create & Activate Conda Env
```shell
conda create -n "CGAN_CAPTCHA_SOLVER" python=3.9.12
conda activate CGAN_CAPTCHA_SOLVER
```
##### Step2: Install PIP Requirements 
```shell
pip install -r requirement.txt
```
##### Step3: Configure captcha_setting.py
##### Step4: Prepare dataset
If you have a lot of different styles in your CAPTCHA dataset, you can use the clustering algorithm:
```shell
python -m src.Clustering.clustering --dataset path_to_dataset
```
##### Step5: Train CAPTCHA generator
if you have quite a few original data, you can generate synthetic CAPTCHA:
```shell
python -m src.GAN.train --dataset_folder --symbols --model_name --saved_model_name
```
##### Step6: Generate as much CAPTCHA as you need for training solver
```shell
python -m src.GAN.create_dataset --dataset_folder --count
```
##### Step7: Train CAPTCHA solver
if you have quite a few original data, you can generate synthetic CAPTCHA:
```shell
python -m src.CNN.train --gen_data --num_gen_train --num_gen_test --saved_model_name --orig_data --num_orig_train --num_orig_test --model_name --saved_model_name'
```
## Metrics

Synthetic CAPTCHA

    Time: 385 seconds to solve 5000 CAPTCHAs

    Accuracy: ~99%

Real CAPTCHA

    Time: 8 seconds to solve 100 CAPTCHAs

    Accuracy:  ~65%

## Documentation

> You can check some details about this solver in the [docs](https://github.com/Alexander-Zadorozhnyy/Licence-Plate-Recognition/docs) directory:
> - docs/report.pdf - educational practice's report (RU)

## Authors

- [@ZadorozhnyyA](https://github.com/Alexander-Zadorozhnyy)

## License

Source code of this repository is released under
the [Apache-2.0 license](https://choosealicense.com/licenses/apache-2.0/)