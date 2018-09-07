#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 15:36:15 2018

@author: yang
"""

from keras.models import Model, model_from_json
from keras.layers.core import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.merge import Concatenate
from keras.layers import Input
from keras.callbacks import ModelCheckpoint
import config as cfg


class SegNet(object):

    #初始化
    def __init__(self):
        self.BATCH_SIZE = cfg.BATCH_SIZE
        self.NUM_EPOCHS = cfg.NUM_EPOCHS
        self.MODEL_PATH = cfg.MODEL_PATH
        self.CHECKPOINT_DIR = cfg.CHECKPOINT_DIR
        self.OPTIMIZER = cfg.OPTIMIZER
        self.LOSS = cfg.LOSS
        self.METRICS = cfg.METRICS
        self.model = None


    #构建模型结构(SegNet结构+Concatenate）
    def build_SegNet(self, num_classes):

        # input
        # input_data = Input(shape=(384, 384, 3))
        input_data = Input(shape=cfg.INPUT_SIZE)

        # convolution block 1
        x_conv1_1 = Conv2D(filters=64, kernel_size=3, padding='same', name='conv1_1', activation='relu')(input_data)
        x_conv1_2 = Conv2D(filters=64, kernel_size=3, padding='same', name='conv1_2', activation='relu')(x_conv1_1)
        x_pooling1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling1')(x_conv1_2)
        x_dropout1 = Dropout(0.25, name='dropout1')(x_pooling1)

        # convolution block 2
        x_conv2_1 = Conv2D(filters=128, kernel_size=3, padding='same', name='conv2_1', activation='relu')(x_dropout1)
        x_conv2_2 = Conv2D(filters=128, kernel_size=3, padding='same', name='conv2_2', activation='relu')(x_conv2_1)
        x_pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling2')(x_conv2_2)
        x_dropout2 = Dropout(0.25, name='dropout2')(x_pooling2)

        # convolution block 3
        x_conv3_1 = Conv2D(filters=256, kernel_size=3, padding='same', name='conv3_1', activation='relu')(x_dropout2)
        x_conv3_2 = Conv2D(filters=256, kernel_size=3, padding='same', name='conv3_2', activation='relu')(x_conv3_1)
        x_conv3_3 = Conv2D(filters=256, kernel_size=3, padding='same', name='conv3_3', activation='relu')(x_conv3_2)
        x_pooling3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling3')(x_conv3_3)
        x_dropout3 = Dropout(0.25, name='dropout3')(x_pooling3)

        # convolution block 4
        x_conv4_1 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv4_1', activation='relu')(x_dropout3)
        x_conv4_2 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv4_2', activation='relu')(x_conv4_1)
        x_conv4_3 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv4_3', activation='relu')(x_conv4_2)
        x_pooling4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling4')(x_conv4_3)
        x_dropout4 = Dropout(0.25, name='dropout4')(x_pooling4)

        # convolution block 5
        x_conv5_1 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv5_1', activation='relu')(x_dropout4)
        x_conv5_2 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv5_2', activation='relu')(x_conv5_1)
        x_conv5_3 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv5_3', activation='relu')(x_conv5_2)
        x_pooling5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pooling5')(x_conv5_3)
        x_dropout5 = Dropout(0.25, name='dropout5')(x_pooling5)

        # convolution block 6 1x1
        x_conv6_1 = Conv2D(filters=512, kernel_size=1, padding='same', name='conv6_1', activation='relu')(x_dropout5)
        x_conv6_2 = Conv2D(filters=512, kernel_size=1, padding='same', name='conv6_2', activation='relu')(x_conv6_1)
        x_conv6_3 = Conv2D(filters=512, kernel_size=1, padding='same', name='conv6_3', activation='relu')(x_conv6_2)

        # upsampling block 7
        x_up7 = UpSampling2D(size=(2, 2), name='upsampling7')(x_conv6_3)
        x_concat7 = Concatenate(axis=-1, name='concatenate7')([x_up7, x_conv5_3])
        x_conv7_1 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv7_1', activation='relu')(x_concat7)
        x_conv7_2 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv7_2', activation='relu')(x_conv7_1)
        x_conv7_3 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv7_3', activation='relu')(x_conv7_2)

        # upsampling block 8
        x_up8 = UpSampling2D(size=(2, 2), name='upsampling8')(x_conv7_3)
        x_concat8 = Concatenate(axis=-1, name='concatenate8')([x_up8, x_conv4_3])
        x_conv8_1 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv8_1', activation='relu')(x_concat8)
        x_conv8_2 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv8_2', activation='relu')(x_conv8_1)
        x_conv8_3 = Conv2D(filters=512, kernel_size=3, padding='same', name='conv8_3', activation='relu')(x_conv8_2)

        # upsampling block 9
        x_up9 = UpSampling2D(size=(2, 2), name='upsampling9')(x_conv8_3)
        x_concat9 = Concatenate(axis=-1, name='concatenate9')([x_up9, x_conv3_3])
        x_conv9_1 = Conv2D(filters=256, kernel_size=3, padding='same', name='conv9_1', activation='relu')(x_concat9)
        x_conv9_2 = Conv2D(filters=256, kernel_size=3, padding='same', name='conv9_2', activation='relu')(x_conv9_1)

        # upsampling block 10
        x_up10 = UpSampling2D(size=(2, 2), name='upsampling10')(x_conv9_2)
        x_concat10 = Concatenate(axis=-1, name='concatenate10')([x_up10, x_conv2_2])
        x_conv10_1 = Conv2D(filters=128, kernel_size=3, padding='same', name='conv10_1', activation='relu')(x_concat10)
        x_conv10_2 = Conv2D(filters=128, kernel_size=3, padding='same', name='conv10_2', activation='relu')(x_conv10_1)

        # upsampling block 11
        x_up11 = UpSampling2D(size=(2, 2), name='upsampling11')(x_conv10_2)
        x_concat11 = Concatenate(axis=-1, name='concatenate11')([x_up11, x_conv1_2])
        x_conv11_1 = Conv2D(filters=64, kernel_size=3, padding='same', name='conv11_1', activation='relu')(x_concat11)
        x_conv11_2 = Conv2D(filters=64, kernel_size=3, padding='same', name='conv11_2', activation='relu')(x_conv11_1)

        # output layer
        prediction = Conv2D(filters=num_classes, kernel_size=1, padding='same', name='output', activation='softmax')(x_conv11_2)

        self.model = Model(inputs=input_data, outputs=prediction)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)

    #模型加载
    def load_model(self):
        self.model = model_from_json(open(cfg.MODEL_PATH).read())
        self.model.load_weights(cfg.WEIGHTS_PATH)
        self.model.compile(optimizer=self.OPTIMIZER, loss=self.LOSS, metrics=self.METRICS)

    #模型保存
    def save_model(self):
        json_string = self.model.to_json()
        open(cfg.MODEL_PATH, 'w').write(json_string)
        self.model.save_weights(cfg.WEIGHTS_PATH, overwrite=True)

    #模型训练
    def train(self, x_train, y_train):
        checkpoint = ModelCheckpoint(filepath=cfg.CHECKPOINT_DIR + "model-{epoch:02d}.h5", monitor='val_acc')
        self.model.fit(x_train, y_train, batch_size=cfg.BATCH_SIZE, epochs=cfg.NUM_EPOCHS,
                      verbose=cfg.VERBOSE, validation_split=0.1, callbacks=[checkpoint])
        # self.model.fit(x_train, y_train, batch_size=cfg.BATCH_SIZE, epochs=cfg.NUM_EPOCHS,
        #                verbose=cfg.VERBOSE, validation_data=[x_val, y_val], callbacks=[checkpoint])

    #模型再训练
    def continue_train(self, x_train, y_train):
        self.load_model()
        checkpoint = ModelCheckpoint(filepath=cfg.CHECKPOINT_DIR + "model-{epoch:02d}.h5", monitor='val_acc')
        self.model.fit(x_train, y_train, batch_size=cfg.BATCH_SIZE, epochs=cfg.NUM_EPOCHS,
              verbose=cfg.VERBOSE, validation_split=0.1, callbacks=[checkpoint])
        # self.model.fit(x_train, y_train, batch_size=cfg.BATCH_SIZE, epochs=cfg.NUM_EPOCHS,
        #                verbose=cfg.VERBOSE, validation_data=[x_val, y_val], callbacks=[checkpoint])

    #测试
    def test(self, x_test, y_test):
        self.load_model()
        self.model.evaluate(x_test, y=y_test, batch_size=cfg.BATCH_SIZE, verbose=cfg.VERBOSE)

    #预测
    def predict(self, input_image):
        self.load_model()
        precition = self.model.predict(input_image, verbose=0)
        return precition
