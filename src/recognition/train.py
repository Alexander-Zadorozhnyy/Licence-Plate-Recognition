import argparse
import os

import tensorflow as tf
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.tensorflow.metrics import CERMetric, WERMetric
from mltu.tensorflow.callbacks import TrainLogger, Model2onnx
from mltu.tensorflow.losses import CTCloss

from dataset.dataloader import Dataloader
from config import WIDTH, HEIGHT, EPOCH, CHANNELS, START_LR, SAVE_MODEL_PATH, SYMBOLS, BATCH
from model.model import MRNET


def get_parser_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train_path', type=str,
                        default='train_data_root', help='root path to dataset')
    parser.add_argument('--valid_path', type=str,
                        default='valid_data_root', help='root path where to save model')
    parser.add_argument('--augment', type=bool,
                        default=False, help='enable data augmentation or not')
    parser.add_argument('--saved_model_path', type=str,
                        default=None, help='root path to saved_model')
    parser.add_argument('--save_csv', type=str,
                        default=None, help='save train and valid data dataset as csv')
    return vars(parser.parse_args())


def main(train_path, valid_path, augment, saved_model_name, save_csv):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    dataloader = Dataloader(train_path=train_path,
                            val_path=valid_path,
                            symbols=SYMBOLS,
                            augment=augment)
    dataloader.build(image_width=WIDTH, image_height=HEIGHT, batch=BATCH)
    dataloader.print_statistic()

    model = MRNET(symbol_count=dataloader.get_num_characters() + 1,
                  input_shape=(HEIGHT, WIDTH, CHANNELS))
    model.build()

    if saved_model_name is not None:
        model.load(saved_model_name)

    # Optimizer
    opt = Adam(learning_rate=START_LR)

    # Compile the model and return
    model.compile(opt=opt,
                  loss=CTCloss(),
                  metrics=[
                      CERMetric(vocabulary=dataloader.get_str_characters()),
                      WERMetric(vocabulary=dataloader.get_str_characters())
                  ],
                  )

    # Define callbacks
    # earlystopper = EarlyStopping(monitor='val_WER', patience=10, verbose=1)
    checkpoint = ModelCheckpoint(f"{SAVE_MODEL_PATH}/model.h5",
                                 monitor='val_WER',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='min')
    train_logger = TrainLogger(f"{SAVE_MODEL_PATH}")
    tb_callback = TensorBoard(f'{SAVE_MODEL_PATH}/logs', update_freq=1)
    reduce_lr_on_plat = ReduceLROnPlateau(monitor='val_WER',
                                          factor=0.9,
                                          min_delta=1e-10,
                                          patience=5,
                                          verbose=1,
                                          mode='auto')
    model2onnx = Model2onnx(f"{SAVE_MODEL_PATH}/model.h5")

    train_dataloader = dataloader.get_train_dataloader()
    valid_dataloader = dataloader.get_valid_dataloader()

    # Train the model
    model.train(train_data=train_dataloader,
                validation_data=valid_dataloader,
                epochs=EPOCH,
                callbacks=[checkpoint, train_logger, tb_callback, reduce_lr_on_plat, model2onnx],
                workers=8,
                )

    model.save(path=SAVE_MODEL_PATH)

    if save_csv:
        # Save training and validation datasets as csv files
        train_dataloader.to_csv(os.path.join(SAVE_MODEL_PATH, 'train.csv'))
        valid_dataloader.to_csv(os.path.join(SAVE_MODEL_PATH, 'val.csv'))


if __name__ == '__main__':
    # main(train_path="../../../datasets/ocr/train",
    #      valid_path="../../../datasets/ocr/valid",
    #      augment=True,
    #      save_csv=False,
    #      saved_model_name=None)
    args = get_parser_args()
    main(train_path=args['train_path'],
         valid_path=args['valid_path'],
         augment=args['augment'],
         saved_model_name=args['saved_model_path'],
         save_csv=args['save_csv'],
         )
