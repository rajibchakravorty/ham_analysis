import numpy as np
from os.path import join
import tensorflow as tf

from tf_parser import Parser

device_string = '/device:GPU:0'
#device_string = '/device:CPU:0'

## definition of epoch in terms of batch number
## 34662/1011/1530 Train/Valid/Test data
batch_size = 32
batch_per_training_epoch = int(np.floor(5*1011/32)) #int(np.floor(7490/32))

## batches to be used during statistics collections
batch_per_validation_epoch = int(np.floor(1530/32)) #int(np.floor(994/32))

learning_rate_info = dict()
learning_rate_info['init_rate'] = 0.0005
learning_rate_info['decay_steps'] = 30 * batch_per_validation_epoch
learning_rate_info['decay_factor'] = 0.95
learning_rate_info['staircase']=True

##loss operations
loss_op=tf.losses.sparse_softmax_cross_entropy
one_hot=False
loss_op_kwargs = None

##optimizers
optimizer = tf.train.AdamOptimizer
optimizer_kwargs = None

image_height = 128
image_width  = 128
image_channel  = 3

class_numbers = 7
class_weights = np.array([21.48717949, 12.85933504, 6.22277228, 5.93624557, 1., 46.12844037, 53.4893617])
class_weights = class_weights[np.newaxis, 1]

checkpoint_path = './checkpoints4'
model_checkpoint_path = join( checkpoint_path, 'model.ckpt')
prior_weights = None
train_summary_path = join( checkpoint_path, 'train' )
valid_summary_path = join( checkpoint_path, 'valid' )


train_tfrecords = '/home/rajib/skin/ham_analysis/record/train.tfrecords'
valid_tfrecords = '/home/rajib/skin/ham_analysis/record/test.tfrecords'

## information for parsing the tfrecord
features = {'image': tf.FixedLenFeature([], tf.string),
            'disease_label': tf.FixedLenFeature([], tf.int64),
            'cancer_label' : tf.FixedLenFeature([], tf.int64),
            'decision_type' : tf.FixedLenFeature([], tf.int64),
            'age' : tf.FixedLenFeature([], tf.float32),
            'sex' : tf.FixedLenFeature([], tf.int64),
            'localization' : tf.FixedLenFeature([], tf.int64)}
train_parser = Parser(features, True, image_height, image_width)
valid_parser = Parser(features, False, image_height, image_width)
