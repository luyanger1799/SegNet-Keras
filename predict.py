import os
from SegNet import SegNet
import util
import config as cfg
import numpy as np
import tensorflow as tf
import helpers
import cv2

test_image = util.load_image(cfg.TEST_IMAGE_PATH)
print(test_image.shape)

with tf.device('/cpu:0'):
    test_image = np.float32(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)

test_image = np.array(test_image)

segnet = SegNet()
segnet.load_model()
prediction = segnet.predict(input_image=test_image)

prediction = helpers.reverse_one_hot(prediction)

# this needs to get generalized
class_names_list, label_values = helpers.get_label_info(os.path.join("CamVid", "class_dict.csv"))

out_vis_image = helpers.colour_code_segmentation(prediction, label_values)

file_path = cfg.TEST_IMAGE_PATH.replace(".png", "_pred.png")
print(file_path)

out_vis_image = out_vis_image.reshape(cfg.INPUT_SIZE)
print(out_vis_image.shape)
cv2.imwrite(file_path, cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
