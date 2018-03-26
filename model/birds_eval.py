"""
TensorFlow script for evaluating Inception-v3 on test data.
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



def list_images(directory):
    """
    Get all the images and labels in directory/label/*.jpg
    """
    labels = [f for f in os.listdir(directory) if
              os.path.isdir(os.path.join(directory, f))]
    labels.sort()
    files_and_labels = []
    for label in labels:
        for f in os.listdir(os.path.join(directory, label)):
            files_and_labels.append((os.path.join(directory, label, f), label))

    filenames, labels = zip(*files_and_labels)
    filenames = list(filenames)
    labels = list(labels)
    unique_labels = sorted(list(set(labels)), key=int)

    label_to_int = {}
    for i, label in enumerate(unique_labels, 1):
        label_to_int[label] = i
    labels = [label_to_int[l] for l in labels]

    return filenames, labels


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


def test(args):
    """Evaluates Inception-v3 on the test dataset."""

    ### Define the filepaths and directories used during evaluation.
    # Directory containing training and validation data:
    test_dir = os.path.join(args['data_dir'], 'test')
    # Location of checkpoint to restore from if `args['restore']` is True:
    restore_dir = args['eval_dir']

    # Get the list of filenames and corresponding list of labels for training
    # and validation. 'num_split' is the number of spectrograms from
    # each class in the training set to divert to the validation set.
    logging.info('Loading dataset from `test_dir`...')
    test_filenames, test_labels = list_images(test_dir)
    logging.info("%d test images over %d classes found." %
                 (len(test_filenames), len(set(test_labels))))
    # Number of classes used by Inception is set of labels plus 1 (dummy class).
    num_classes = len(set(test_labels)) + 1
    top_k = 5

    ### Define the graph.
    logging.info("Building graph...")
    g2 = tf.Graph()
    with g2.as_default():
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

            # (4) Standard preprocessing for Inception-v3 net.
            #     ...scale `0 -> 1` pixel range to `-1 -> 1`
            image = tf.subtract(image, 0.5)
            image = tf.multiply(image, 2.0)
            return image, label

        # Load test data from `test_filenames` then batch the data.
        val_dataset = tf.data.Dataset.from_tensor_slices((test_filenames,
                                                          test_labels))
        val_dataset = val_dataset.map(_parse_function,
                                      num_parallel_calls=args['num_workers'])
        batched_val_dataset = val_dataset.batch(args['batch_size'])

        iterator = tf.contrib.data.Iterator.from_structure(
            batched_val_dataset.output_types,
            batched_val_dataset.output_shapes)
        images, labels = iterator.get_next()

        val_init_op = iterator.make_initializer(batched_val_dataset)
        is_training = tf.placeholder(tf.bool)

        # Load the Inception-v3 model from the Slim library.
        inception = tf.contrib.slim.nets.inception
        with slim.arg_scope(inception.inception_v3_arg_scope(
                            weight_decay=0.00004)):
            logits, end_points = inception.inception_v3(images,
                num_classes=num_classes,
                is_training=is_training,
                dropout_keep_prob=0.8)

        # Restore all weights variables in the model.
        # Calling function `restore_fn(sess)` will load the pretrained weights
        # from the checkpoint file at args['model_path'].
        all_variables = tf.contrib.framework.get_variables_to_restore()
        restore_fn = tf.contrib.framework.assign_from_checkpoint_fn(
            tf.train.latest_checkpoint(restore_dir), all_variables)

        tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                               logits=logits,
                                               weights=1.0)
        tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                               logits=end_points['AuxLogits'],
                                               weights=0.4)
        loss = tf.losses.get_total_loss()

        # Compute some evaluation metrics.
        kw_predictions = tf.argmax(end_points['Predictions'], 1)
        kw_probabilities = end_points['Predictions']
        prediction = tf.to_int32(tf.argmax(logits, 1))
        correct_prediction = tf.equal(prediction, labels)
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        top5_accuracy = tf.reduce_mean(tf.cast(
            tf.nn.in_top_k(end_points['Logits'], labels, top_k), tf.float32))

        # Merge all summary nodes and create an init op.
        init = tf.global_variables_initializer()
        tf.get_default_graph().finalize()


    with tf.Session(graph=g2) as sess:
        sess.run(init)
        logging.info('Restoring from latest checkpoint in %s.', args['eval_dir'])
        restore_fn(sess)
        # Run through all test data once.
        logging.info('Starting evaluation on test split...')
        sess.run(val_init_op)
        total_loss, count = 0, 0
        num_correct, num_samples = 0, 0
        top5_acc = 0
        while True:
            try:
                this_loss, correct_pred, top5_sample_acc = sess.run(
                    [loss, correct_prediction, top5_accuracy],
                    {is_training: False})
                num_correct += correct_pred.sum()
                num_samples += correct_pred.shape[0]
                top5_acc += top5_sample_acc
                total_loss += this_loss
                count += 1
            except tf.errors.OutOfRangeError:
                break

        # Calculate the fraction of images that were correctly classified.
        test_acc = float(num_correct) / num_samples
        # Calculate top-5 accuracy.
        top5_test_acc = float(top5_acc) / count
        # Calculate the average loss.
        test_loss = float(total_loss) / count
        logging.info('Test Loss: %f' % test_loss)
        logging.info('Test Accuracy: %f' % test_acc)
        logging.info('Top-5 Accuracy: %f' % top5_test_acc)
