
import tensorflow as tf

from tensorflow.layers import (conv2d, max_pooling2d, dense, batch_normalization, Flatten)

from tensorflow.nn import relu, leaky_relu, sigmoid


def model(input_image, training):

    conv1 = tf.layers.conv2d(input_image, filters=24, kernel_size=(3, 3), strides=(2, 2),
                             padding='valid', activation = leaky_relu, use_bias=True,
                             kernel_initializer = tf.initializers.glorot_normal(),
                             bias_initializer = tf.initializers.glorot_normal(),
                             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.1),
                             trainable=True, name='conv1')

    max1 = tf.layers.max_pooling2d(conv1, pool_size=(3,3), strides=(2, 2), padding='valid',
                                   name='max1')

    batch_norm1 = batch_normalization(max1, training=training)

    conv2 = tf.layers.conv2d(batch_norm1, filters=36, kernel_size=(3, 3), strides=(2, 2),
                             padding='valid', activation=leaky_relu, use_bias=True,
                             kernel_initializer=tf.initializers.glorot_normal(),
                             bias_initializer=tf.initializers.glorot_normal(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                             trainable=True, name='conv2')

    max2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(1, 1), padding='valid',
                                   name='max2')

    batch_norm2 = batch_normalization(max2, training=training)

    conv3 = tf.layers.conv2d(batch_norm2, filters=60, kernel_size=(2, 2), strides = (1,1),
                             padding='valid', activation = leaky_relu, use_bias=True,
                             kernel_initializer = tf.initializers.glorot_normal(),
                             bias_initializer = tf.initializers.glorot_normal(),
                             kernel_regularizer = tf.contrib.layers.l2_regularizer(0.1),
                             trainable=True, name='conv3')

    max3 = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(1, 1), padding='valid',
                                   name='max3')

    batch_norm3 = batch_normalization(max3, training=training)

    conv4 = tf.layers.conv2d(batch_norm3, filters=96, kernel_size=(2, 2), strides=(1, 1),
                             padding='valid', activation=leaky_relu, use_bias=True,
                             kernel_initializer=tf.initializers.glorot_normal(),
                             bias_initializer=tf.initializers.glorot_normal(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                             trainable=True, name='conv4')

    max4 = tf.layers.max_pooling2d(conv4, pool_size=(2, 2), strides=(1, 1), padding='valid',
                                   name='max4')

    batch_norm4 = batch_normalization(max4, training=training)

    conv5 = tf.layers.conv2d(batch_norm4, filters=128, kernel_size=(2, 2), strides=(1, 1),
                             padding='same', activation=leaky_relu, use_bias=True,
                             kernel_initializer=tf.initializers.glorot_normal(),
                             bias_initializer=tf.initializers.glorot_normal(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                             trainable=True, name='conv5')

    max5 = tf.layers.max_pooling2d(conv5, pool_size=(2, 2), strides=(1, 1), padding='valid',
                                   name='max5')

    batch_norm5 = batch_normalization(max5, training=training)

    conv6 = tf.layers.conv2d(batch_norm5, filters=1024, kernel_size=(1, 1), strides=(1, 1),
                             padding='same', activation=leaky_relu, use_bias=True,
                             kernel_initializer=tf.initializers.glorot_normal(),
                             bias_initializer=tf.initializers.glorot_normal(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                             trainable=True, name='conv6')


    max6 = tf.layers.max_pooling2d(conv6, pool_size=(2, 2), strides=(1, 1), padding='valid',
                                   name='max6')

    batch_norm6 = batch_normalization(max6, training=training)

    batch_norm_flat = tf.layers.Flatten()(batch_norm6)

    dense1 = tf.layers.dense(batch_norm_flat, units=512, use_bias=True,
                             kernel_initializer=tf.initializers.glorot_normal(),
                             activation = tf.nn.relu,
                             bias_initializer=tf.initializers.glorot_normal(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                             bias_regularizer=tf.contrib.layers.l1_regularizer(0.),
                             trainable=True, name='dense1')

    batch_norm5 = batch_normalization(dense1, training=training)

    dense2 = tf.layers.dense(batch_norm5, units=7, use_bias=True,
                             kernel_initializer=tf.initializers.glorot_normal(),
                             activation = tf.nn.sigmoid,
                             bias_initializer=tf.initializers.glorot_normal(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1),
                             bias_regularizer=tf.contrib.layers.l1_regularizer(0.),
                             trainable=True, name='dense2')

    return dense2


