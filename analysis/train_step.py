
import tensorflow as tf


from model import model


def get_training_summaries(average_loss, learning_rate):

    trainable_variables = tf.trainable_variables() #tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    summaries = list()
    for var in trainable_variables:

        summaries.append(tf.summary.histogram(var.op.name, var))

    summaries.append(tf.summary.scalar('average training loss', average_loss))
    summaries.append(tf.summary.scalar(learning_rate.op.name, learning_rate))

    return summaries


def get_validation_summaries(average_loss):

    summaries = list()
    summaries.append(tf.summary.scalar('average_validation_loss', average_loss))

    return summaries


def train_steps(batch_images, batch_labels,
                global_step, device, variable_scope_name, optimizer,
                is_training):

    with tf.device(device):

        with tf.variable_scope(variable_scope_name, reuse=tf.AUTO_REUSE):

            model_output = model(batch_images, is_training)

    #batch_output_one_hot = tf.one_hot(batch_labels, 7)
    loss = tf.losses.sparse_softmax_cross_entropy(batch_labels, model_output)

    regularization_loss = tf.losses.get_regularization_loss()

    total_loss = loss + regularization_loss

    gradient = optimizer.compute_gradients(total_loss)

    apply_op = optimizer.apply_gradients(gradient, global_step=global_step)

    return model_output, loss, regularization_loss, apply_op