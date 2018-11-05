import sys
import numpy as np
import tensorflow as tf
from random import shuffle

from skimage.io import imread
from skimage.transform import resize

from datafold.constants import label_dict

import pickle


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def prepare_tf_records(fold_list, output_file):

    # open the TFRecords file
    writer = tf.python_io.TFRecordWriter(output_file)
    for i in range(len(fold_list)):
        current_item = fold_list[i]

        # Load the image
        image_file = current_item[0]
        disease_label = current_item[1]
        cancer_label = current_item[2]
        dx_type = current_item[3]
        age = current_item[4]
        sex = current_item[5]
        localization = current_item[6]

        # Create a feature
        feature = {'disease_label': _int64_feature(disease_label),
                   'cancer_label' : _int64_feature(cancer_label),
                   'image': _bytes_feature(tf.compat.as_bytes(image_file)),
                   'decision_type' : _int64_feature(dx_type),
                   'age' : _float32_feature(age),
                   'sex' : _int64_feature(sex),
                   'localization' : _int64_feature(localization)}
        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
    sys.stdout.flush()


def balance_data(unbalanced_list):

    total_labels = len(label_dict.keys())
    label_counts = np.zeros((total_labels,))
    for dt in unbalanced_list:
        this_label = dt[1]
        label_counts[this_label] += 1

    max_count = np.max(label_counts)
    print(label_counts)
    label_counts = max_count/label_counts

    data_per_labels = dict()
    for l in range(total_labels):
        data_per_labels[l] = list()

    for dt in unbalanced_list:
        this_label = dt[1]
        data_per_labels[this_label].append(dt)

    balanced_data = list()

    for l in range(total_labels):

        for rep in range(int(label_counts[l])):

            balanced_data = balanced_data + data_per_labels[l]

    shuffle(balanced_data)

    return balanced_data


if __name__ == '__main__':

    fold_file = '../datafold/fold_data.pkl'

    with open(fold_file, 'rb') as f:
        train_fold, validation_fold, test_fold = pickle.load(f)

    balanced_train_fold = balance_data(train_fold)

    print('{0}/{1}/{2} Train/Valid/Test data'.format(len(balanced_train_fold),
                                                     len(validation_fold),
                                                     len(test_fold)))

    prepare_tf_records(balanced_train_fold, 'train.tfrecords')
    prepare_tf_records(test_fold, 'test.tfrecords')
    prepare_tf_records(validation_fold, 'validation.tfrecords')