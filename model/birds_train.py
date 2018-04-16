"""
TensorFlow script for finetuning Inception-v3 on audio spectrogram data.

This code was originally based on an example by Olivier Moindrot for finetuning
a VGG model:
(https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/)

Beyond reorganizing the code, I also made many changes to suit the differences
in model architecture (Inception-v3 vs. VGG) and data type (spectrograms vs.
standard images). For example, I added random cropping of spectrograms in the
time dimension as a form of data augmentation.

I also introduced some important features:

* model saver to save trained model weights
* loss monitoring, in addition to train/val accuracy
* summaries to visualize accuracy/loss as a function of epoch in TensorBoard
* ability to view sample images and associated labels during training, as a
  fun visualization and sanity check to confirm the correctness of class labels
* improved optimizer (Adam vs. simple gradient descent w/ fixed learning rate)
* class balancing (my dataset is somewhat class-imbalanced)
* automatic logging of hyperparameters+results to enable hyperparameter search
* image visualization during evaluation plus random samples of true/predicted
  classes. This helped me make sense of the model's predictions, since the ID
  of a birdsong spectrogram isn't obvious to the average human observer.
"""

from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import pickle
from subprocess import call
from shutil import copyfile
from scipy.io import wavfile
import scipy.misc
import librosa
import time
import matplotlib.pyplot as plt
import argparse
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets
import logging



def list_images_split(directory, num_split=2):
    """
    Gets all images in directory/label/*.jpg with corresponding labels.
    """
    labels = [f for f in os.listdir(directory) if
              os.path.isdir(os.path.join(directory, f))]
    labels.sort()
    train_files_and_labels = []
    val_files_and_labels = []

    balance_factors_path = os.path.join(directory, 'balance_factors.txt')
    if os.path.isfile(balance_factors_path):
        logging.info('Class-balancing train data using balance_factors.txt...')
        balance_factors = read_counts_file(balance_factors_path)
        for label in labels:
            for count, f in enumerate(
                    reversed(os.listdir(os.path.join(directory, label)))):
                if count < num_split:
                    val_files_and_labels.append(
                        (os.path.join(directory, label, f), label))
                else:
                    for i in range(balance_factors[label]):
                        train_files_and_labels.append(
                            (os.path.join(directory, label, f), label))
    else:
        logging.info('File balance_factors.txt not found, single pass only.')
        for label in labels:
            for count, f in enumerate(
                    reversed(os.listdir(os.path.join(directory, label)))):
                if count < num_split:
                    val_files_and_labels.append(
                        (os.path.join(directory, label, f), label))
                else:
                    train_files_and_labels.append(
                        (os.path.join(directory, label, f), label))

    train_filenames, train_labels = zip(*train_files_and_labels)
    train_filenames = list(train_filenames)
    train_labels = list(train_labels)
    unique_labels = sorted(list(set(train_labels)), key=int)
    label_to_int = {}
    for i, label in enumerate(unique_labels, 1):
        label_to_int[label] = i
    train_labels = [label_to_int[l] for l in train_labels]

    val_filenames, val_labels = zip(*val_files_and_labels)
    val_filenames = list(val_filenames)
    val_labels = list(val_labels)
    unique_labels = sorted(list(set(val_labels)), key=int)
    label_to_int = {}
    for i, label in enumerate(unique_labels, 1):
        label_to_int[label] = i
    val_labels = [label_to_int[l] for l in val_labels]

    return train_filenames, train_labels, val_filenames, val_labels


def read_label_file(path):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
        dataset_dir: The directory in which the labels file is found.
        filename: The filename where the class names are written.

    Returns:
        A map from a label (integer) to class name.
    """
    labels_filename = path
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index+1:]
    return labels_to_class_names


def write_params_file(path, params):
    """Writes a file with the list of class names.

    Args:
        path:   Params filepath.
        params: Map from parameter names to values.
    """
    with tf.gfile.Open(path, 'w') as f:
        for name, value in sorted(params.items()):
            f.write('%s %s\n' % (name, value))


def read_params_file(path):
    params = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split()
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val
    return params


def read_counts_file(path):
    counts = {}
    with open(path) as f:
        for line in f:
            (key, val) = line.split()
            counts[key] = int(val)
    return counts


def probs_to_ranks(array):
    """
    Returns ranks for each entry in an array.
    E.g. the highest scoring element would receive rank 1, then rank 2, etc.
    """
    output = [0] * len(array)
    for i, x in enumerate(sorted(range(len(array)),
                          key=lambda y: array[y], reverse=True)):
        output[x] = i + 1
    return output


def finetune(args):
    """Finetunes Inception-v3 on labeled spectrograms."""

    train_dir = os.path.join(args['data_dir'], 'train')
    save_dir = os.path.join(args['run_dir'], 'save')          # checkpoints
    summary_dir = os.path.join(args['run_dir'], 'summary')    # logs/summaries
    params_path = os.path.join(args['run_dir'], 'params.txt') # hyperparameters

    summary_dir_logits = os.path.join(summary_dir, 'logits')
    summary_dir_full = os.path.join(summary_dir, 'full')

    for directory in [save_dir, summary_dir_logits, summary_dir_full]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    save_path = os.path.join(save_dir, 'full')

    # Find directory or path containing checkpoint to restore.
    restore_path = args['restore_path']
    restore_dir = args['restore_dir']
    if args['restore_path'] is not None:
        use_restore_path = True
    else:
        use_restore_path = False
        if restore_dir is None:     # Defaults to save directory of current run
            restore_dir = save_dir

    # Get the list of filenames and corresponding list of labels for training
    # and validation. `num_split` is the number of spectrograms from each class
    # that are diverted to the validation set.
    logging.info('Loading dataset from %s...' % train_dir)
    train_filenames, train_labels, val_filenames, val_labels = \
        list_images_split(train_dir, num_split=2)
    num_train = len(train_filenames)
    num_val = len(val_filenames)
    logging.info("%d train images and %d validation images found." %
                 (num_train, num_val))

    num_classes = len(set(train_labels)) + 1    # Need to add 1 for dummy class.

    logging.info("Building graph...")
    g1 = tf.Graph()
    with g1.as_default():
        # Preprocessing for Inception-v3.
        def _parse_function(filename, label):
            # Preprocessing steps:
            # (1) Decode the image from jpg format.
            image_string = tf.read_file(filename)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)
            image.set_shape([224, 341, 3])

            # (2) Resize the image 299x319.
            #     Want a random 3.75 second slice of 4.00 second spectrogram to
            #     be square (299x299) - set new_width accordingly.
            new_height = tf.to_int32(299.0)
            new_width = tf.to_int32(319.0)   # use 341.0 instead for 3.5 s slice
            crop_height = tf.to_int32(299.0)
            crop_width = tf.to_int32(299.0)
            image = tf.image.resize_images(image, [new_height, new_width])

            # (3) Take a random 299x299 crop of the image (random time slice).
            max_offset_height = tf.reshape(new_height - crop_height + 1, [])
            max_offset_width = tf.reshape(new_width - crop_width + 1, [])
            offset_height = tf.constant(0, dtype=tf.int32)
            offset_width = tf.random_uniform([], maxval=max_offset_width, dtype=tf.int32)
            original_shape = tf.shape(image)
            cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
            offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
            image = tf.slice(image, offsets, cropped_shape)
            image.set_shape([299, 299, 3])

            # (4) Standard preprocessing for Inception-v3 net:
            #     Scale `0 -> 1` range for each pixel to `-1 -> 1`.
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            return image, label

        train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames,
                                                            train_labels))
        train_dataset = train_dataset.shuffle(buffer_size=len(train_filenames))
        train_dataset = train_dataset.map(_parse_function,
                                          num_parallel_calls=args['num_workers'])
        batched_train_dataset = train_dataset.batch(args['batch_size'])
        val_dataset = tf.data.Dataset.from_tensor_slices((val_filenames,
                                                          val_labels))
        val_dataset = val_dataset.map(_parse_function,
                                      num_parallel_calls=args['num_workers'])
        batched_val_dataset = val_dataset.batch(args['batch_size'])

        # Define an iterator that can operate on either dataset.
        iterator = tf.contrib.data.Iterator.from_structure(
            batched_train_dataset.output_types,
            batched_train_dataset.output_shapes)
        images, labels = iterator.get_next()

        ### Can add a sample image and corresponding label to the summary as a
        ### sanity check, if desired.
        # tf.summary.image('sample_image',
        #                  tf.reshape(tf.gather(images, 0), [1,299,299,3]))
        # tf.summary.scalar('sample_label', tf.gather(labels, 0))

        # Use these ops to initialize the iterator for the desired dataset.
        train_init_op = iterator.make_initializer(batched_train_dataset)
        val_init_op = iterator.make_initializer(batched_val_dataset)

        is_training = tf.placeholder(tf.bool)

        # Load the Inception-v3 model from the Slim library.
        inception = tf.contrib.slim.nets.inception
        with slim.arg_scope(inception.inception_v3_arg_scope(
                            weight_decay=args['weight_decay'])):
            logits, end_points = inception.inception_v3(images,
                num_classes=num_classes,
                is_training=is_training,
                dropout_keep_prob=args['dropout_keep_prob'])


        # Set epoch counters for the FC-only and full-net training phases:
        logits_epoch = tf.Variable(0, trainable=False, name='logits_epoch')
        reset_logits_epoch = tf.assign(logits_epoch, 0)
        increment_logits_epoch = tf.assign_add(logits_epoch, 1,
                                               name='increment_logits_epoch')
        full_epoch = tf.Variable(0, trainable=False, name='full_epoch')
        reset_full_epoch = tf.assign(full_epoch, 0)
        increment_full_epoch = tf.assign_add(full_epoch, 1,
                                             name='increment_full_epoch')

        # Restore only the layers before Logits/AuxLogits.
        # Calling `init_fn(sess)` will load the pretrained weights from the
        # checkpoint file at args['init_path'].
        layers_exclude = ['InceptionV3/Logits', 'InceptionV3/AuxLogits']
        ## layers_exclude += ['logits_epoch', 'full_epoch']
        variables_to_restore = tf.contrib.framework.get_variables_to_restore(
            exclude=layers_exclude)
        init_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            args['init_path'], variables_to_restore)

        # Use `logits_init` to initialize the final fully-connected layer.
        logits_variables = tf.contrib.framework.get_variables('InceptionV3/Logits')
        logits_variables += tf.contrib.framework.get_variables('InceptionV3/AuxLogits')
        logits_init = tf.variables_initializer(logits_variables)

        # Restore all weights variables in the model from the latest checkpoint
        # in a directory (`restore_dir_fn`) or a filepath (`restore_path_fn`).
        all_variables = tf.contrib.framework.get_variables_to_restore()
        restore_dir_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(restore_dir), all_variables)
        restore_path_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            restore_path, all_variables)

        # Define losses (Logits and AuxLogits).
        tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                               logits=logits,
                                               weights=1.0)
        tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                               logits=end_points['AuxLogits'],
                                               weights=0.4)
        loss = tf.losses.get_total_loss()
        tf.summary.scalar('loss', loss)

        # Use this optimizer to train only the re-initialized final (FC) layer.
        logits_optimizer = tf.train.AdamOptimizer(
            learning_rate=args['learning_rate'], epsilon=args['epsilon'])
        logits_train_op = logits_optimizer.minimize(
            loss, var_list=logits_variables)

        # Use this optimizer to train all model layers.
        full_optimizer = tf.train.AdamOptimizer(
            learning_rate=args['learning_rate'], epsilon=args['epsilon'])
        full_train_op = full_optimizer.minimize(loss)

        # Compute some evaluation metrics.
        kw_predictions = tf.argmax(end_points['Predictions'], 1)
        kw_probabilities = end_points['Predictions']
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=500)
        tf.get_default_graph().finalize()


    with tf.Session(graph=g1) as sess:
        sess.run(init)
        if args['initialize']:     # initialize pre-FC weights using the
                                   # pretrained ImageNet weights
            logging.info('Restoring model (except final layer) from %s...' %
                         args['init_path'])
            init_fn(sess)
            sess.run(logits_init)  # initialize the new FC layer
        elif use_restore_path:
            logging.info('Restoring model from %s...' % restore_path)
            restore_path_fn(sess)
        else:
            logging.info('Restoring model from latest checkpoint in %s...' %
                         restore_dir)
            restore_dir_fn(sess)

        summary_writer_logits = tf.summary.FileWriter(summary_dir_logits,
                                                      sess.graph)
        summary_writer_full = tf.summary.FileWriter(summary_dir_full,
                                                    sess.graph)

        logits_ep, full_ep = sess.run([logits_epoch, full_epoch])
        current_ep = logits_ep + full_ep
        total_ep = current_ep + args['num_epochs']

        for epoch in range(current_ep, total_ep):
            logging.info('Starting epoch %d / %d' % (epoch, total_ep - 1))
            sess.run(train_init_op)
            steps_per_epoch = int(num_train // args['batch_size'])

            for i in range(steps_per_epoch):
                global_step = epoch * steps_per_epoch + i
                try:
                    if args['freeze_conv_layers']:
                        _, _, summary_logits = sess.run(
                            [logits_train_op, accuracy, merged],
                            {is_training: True})
                        summary_writer_logits.add_summary(summary_logits,
                                                          global_step)
                    else:
                        _, _, summary_full = sess.run(
                            [full_train_op, accuracy, merged],
                            {is_training: True})
                        summary_writer_full.add_summary(summary_full,
                                                        global_step)
                except tf.errors.OutOfRangeError:
                    logging.info('OutOfRangeError during training.')
                    break

            # Check accuracy on the training and validation sets.
            if epoch % 10 == 0 or epoch == total_ep - 1:
                # Check training accuracy and loss.
                sess.run(train_init_op)
                num_correct, num_samples, total_loss, count = 0, 0, 0, 0
                for i in range(steps_per_epoch):
                    try:
                        this_loss, correct_pred = sess.run(
                            [loss, correct_prediction],
                            {is_training: False})
                        num_correct += correct_pred.sum()
                        num_samples += correct_pred.shape[0]
                        total_loss += this_loss
                        count += 1
                    except tf.errors.OutOfRangeError:
                        logging.info('OutOfRangeError while checking train acc.')
                        break
                # Find the fraction of spectrograms that were correctly classified.
                train_acc = float(num_correct) / num_samples
                # Calculate the average loss.
                train_loss = float(total_loss) / count

                # Check validation accuracy and loss.
                sess.run(val_init_op)
                num_correct, num_samples = 0, 0
                for i in range(int(num_val // args['batch_size'])):
                    try:
                        correct_pred = sess.run(correct_prediction,
                                                {is_training: False})
                        num_correct += correct_pred.sum()
                        num_samples += correct_pred.shape[0]
                    except tf.errors.OutOfRangeError:
                        logging.info('OutOfRangeError while checking val acc.')
                        break
                # Find the fraction of spectrograms that were correctly classified.
                val_acc = float(num_correct) / num_samples
                logging.info('Train Loss: %f' % train_loss)
                logging.info('Train Accuracy: %f' % train_acc)
                logging.info('Val Accuracy: %f' % val_acc)

            # Increment the appropriate epoch counter.
            if args['freeze_conv_layers']:
                sess.run(increment_logits_epoch)
            else:
                sess.run(increment_full_epoch)
            logits_ep, full_ep = sess.run([logits_epoch, full_epoch])
            # Save the model every 20 epochs or on the final epoch.
            if epoch % 20 == 0 or epoch == total_ep - 1:
                logging.info('Saving model to %s.' % save_path)
                saver.save(sess, save_path, global_step=epoch)

        ### End of session. ###

        # Save this run's hyperparameters and final results (loss, train and
        # validation accuracies) to a params file.
        params = {}
        params['run_dir'] = args['run_dir']
        params['logits_epochs'] = logits_ep
        params['full_epochs'] = full_ep
        params['learning_rate'] = args['learning_rate']
        params['freeze_conv_layers'] = args['freeze_conv_layers']
        params['epsilon'] = args['epsilon']
        params['loss'] = train_loss
        params['train_accuracy'] = train_acc
        params['val_accuracy'] = val_acc
        write_params_file(params_path, params)
        logging.info("Parameters and results saved in params file: %s" %
                     params_path)

        # Record lists of trainable variables and regularization losses to
        # verify that all components were correctly included in training.
        losses_path = os.path.join(args['run_dir'], 'regularization_losses.txt')
        train_vars_path = os.path.join(args['run_dir'], 'train_variables.txt')
        global_vars_path = os.path.join(args['run_dir'], 'global_variables.txt')
        with tf.gfile.Open(losses_path, 'w') as f:
            for item in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                f.write('%s\n' % item.name)
        with tf.gfile.Open(train_vars_path, 'w') as f:
            for item in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
                f.write('%s\n' % item.name)
        with tf.gfile.Open(global_vars_path, 'w') as f:
            for item in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
                f.write('%s\n' % item.name)
