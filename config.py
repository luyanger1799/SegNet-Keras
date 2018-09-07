
import os

#configuration

BATCH_SIZE = 5
NUM_EPOCHS = 500
VERBOSE = 1
CHECKPOINT_DIR = "checkpoint/"
MODEL_PATH = "model/model.json"
WEIGHTS_PATH = "model/weights.h5"
PREDICTION_DIR = "predition/"

#网络配置属性
OPTIMIZER = 'adam'
LOSS = 'categorical_crossentropy'
METRICS = ['accuracy']

DATASET_DIR = "CamVid"

TEST_IMAGE_PATH = "prediction/Seq05VD_f03270.png"

INPUT_SIZE = (320, 320, 3)
IMAGE_SIZE = (320, 320)