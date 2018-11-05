
import numpy as np
import tensorflow as tf

from tfread import prepare_dataset

from train_step import (train_steps, get_training_summaries,
                        get_validation_summaries)

from constants import (train_tfrecords, valid_tfrecords,
                       train_parser, valid_parser,
                       batch_size, batch_per_training_epoch, device_string,
                       learning_rate_info, image_height, image_width,
                       image_channel, batch_per_validation_epoch, prior_weights,
                       train_summary_path, valid_summary_path,
                       model_checkpoint_path)

if __name__ == '__main__':

    default_graph = tf.get_default_graph()

    with default_graph.as_default() as graph:
        config=tf.ConfigProto()    
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=graph)

        global_step = tf.train.create_global_step()
        learning_rate = tf.train.exponential_decay(learning_rate_info['init_rate'],
                                                   global_step,
                                                   learning_rate_info['decay_steps'],
                                                   learning_rate_info['decay_factor'],
                                                   staircase=learning_rate_info['staircase'],
                                                   name='learning_rate')
        is_training = tf.placeholder(dtype=tf.bool, name='is_training')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        training_dataset = prepare_dataset(train_tfrecords, train_parser.parse_example,
                                           batch_size)
        training_iterator = training_dataset.make_one_shot_iterator()


        validation_dataset = prepare_dataset(valid_tfrecords, valid_parser.parse_example,
                                           batch_size)
        validation_iterator = validation_dataset.make_one_shot_iterator()

        handle = tf.placeholder(tf.string, shape=[], name='data_handler')
        iterator = tf.data.Iterator.from_string_handle(
            handle, training_dataset.output_types, training_dataset.output_shapes)
        next_element = iterator.get_next()

        input_image = tf.placeholder(dtype=tf.float32,
                                     shape=[None, image_height, image_width, image_channel],
                                     name='image_input')
        input_labels = tf.placeholder(dtype=tf.int64,
                                      shape=[None, ],
                                      name='label_input')

        model_classes, loss, regu_loss, apply = train_steps(input_image, input_labels, global_step,
                                                            device_string, 'model',
                                                            optimizer, is_training)

        average_loss = tf.placeholder(dtype=tf.float32, shape=[], name='average_loss')
        current_learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='current_learning_rate')
        train_summaries = get_training_summaries(average_loss, current_learning_rate)
        validation_summaries = get_validation_summaries(average_loss)
        training_summary_op = tf.summary.merge(train_summaries)
        validation_summary_op = tf.summary.merge(validation_summaries)

        sess.run(tf.global_variables_initializer())
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())

        saver = tf.train.Saver(max_to_keep=20, pad_step_number=True,
                               keep_checkpoint_every_n_hours=1)

        if prior_weights is not None:
            saver.restore(sess, prior_weights)
            current_step, l_rate = sess.run([global_step, learning_rate])
            print('Starting from Step {0} and learning rate {1}'.format(current_step, l_rate))

        train_summary_writer = tf.summary.FileWriter(logdir=train_summary_path, graph=sess.graph)
        valid_summary_writer = tf.summary.FileWriter(logdir=valid_summary_path, graph=sess.graph)

        # Loop forever, alternating between training and validation.
        while True:
            current_training = sess.run(is_training, feed_dict={is_training: True})
            for batch in range(batch_per_training_epoch):
                current_step = sess.run(global_step)
                print('Step Number {0}'.format(current_step))
                im, label = sess.run(next_element, feed_dict={handle: training_handle})
                sess.run(apply, feed_dict={input_image: im,
                                           input_labels: label,
                                           is_training: True})

            current_training = sess.run(is_training, feed_dict={is_training: False})
            current_step = sess.run(global_step)

            mean_training_loss = 0.
            image_number = 0
            for batch in range(batch_per_validation_epoch):
                im, label = sess.run(next_element, feed_dict={handle: training_handle})
                image_number += im.shape[0]
                l = sess.run(loss, feed_dict={input_image: im,
                                              input_labels: label,
                                              is_training: False})

                mean_training_loss += l

            mean_training_loss /= image_number
            mean_validation_loss = 0.
            image_number = 0
            for batch in range(batch_per_validation_epoch):
                im, label = sess.run(next_element, feed_dict={handle: validation_handle})
                print(im.shape)
                image_number += im.shape[0]
                l = sess.run(loss, feed_dict={input_image: im,
                                             input_labels: label,
                                             is_training: False})

                mean_validation_loss += l

            mean_validation_loss /= image_number

            learn_rate = sess.run(learning_rate)
            training_summaries = sess.run(training_summary_op, feed_dict={average_loss: mean_training_loss,
                                                                          current_learning_rate: learn_rate})
            train_summary_writer.add_summary(training_summaries, global_step=current_step)

            validation_summaries = sess.run(validation_summary_op, feed_dict={average_loss: mean_validation_loss})
            valid_summary_writer.add_summary(validation_summaries, global_step=current_step)

            train_summary_writer.flush()
            valid_summary_writer.flush()

            print('Step {2}: Mean training/validation loss {0}/{1}'.format(mean_training_loss,
                                                                           mean_validation_loss,
                                                                           current_step))

            saver.save(sess, model_checkpoint_path, global_step=current_step)
