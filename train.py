from keras import backend as K
import tensorflow as tf
from SegNet import SegNet
import util
import os
import numpy as np
import helpers
import config as cfg

K.tensorflow_backend._get_available_gpus()

# this needs to get generalized
class_names_list, label_values = helpers.get_label_info(os.path.join("CamVid", "class_dict.csv"))

num_classes = len(label_values)

# Load the data
print("Loading the data ...")
train_input_names, train_output_names, test_input_names, test_output_names = util.prepare_data(cfg.DATASET_DIR)

#id_list = np.random.permutation(len(train_input_names))
input_data = []
output_labels = []
val_data = []
val_labels =[]


for img_name in train_input_names:
    input_image = util.load_image(img_name)
    with tf.device('/cpu:0'):
        input_image = np.float32(input_image) / 255.0

        #input_data.append(np.expand_dims(input_image, axis=0))
        input_data.append(input_image)
        print(img_name)

for labels_name in train_output_names:
    output_image = util.load_image(labels_name)
    with tf.device('/cpu:0'):
        output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))

        #output_labels.append(np.expand_dims(output_image, axis=0))
        output_labels.append(output_image)
        print(labels_name)


input_data = np.array(input_data)
output_labels = np.array(output_labels)
val_data = np.array(val_data)
val_labels = np.array(val_labels)

print("Finish loading the data. ")

#K.set_image_dim_ordering('tf')

print("Start training...")
segnet = SegNet()
segnet.build_SegNet(num_classes)
#segnet.train(x_train=input_data, y_train=output_labels)
segnet.continue_train(x_train=input_data, y_train=output_labels)
print("Finish training.")

print("Save the model...")
segnet.save_model()

print("Finish all")

