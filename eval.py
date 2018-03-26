"""
TensorFlow script for evaluating a trained Inception-v3 net on spectrogram data.
"""

from __future__ import division
from __future__ import print_function

from model import birds_eval

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


parser = argparse.ArgumentParser(description='Inception-v3 Training')
parser.add_argument('--data_dir', default='data',
                    help="Directory containing `train/`, `test/`, `labels.txt`.")
parser.add_argument('--eval_dir', default='runs/0/save/full',
                    help="Location of model checkpoint to be evaluated.")
parser.add_argument('--batch_size', default=8, type=int,
                    help="Batch size for evaluation.")
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers.")
parser.add_argument('--dropout_keep_prob', default=0.8, type=float,
                    help="Dropout keep_prob.")
parser.add_argument('--weight_decay', default=4e-5, type=float,
                    help="Weight decay for weights variables.")
args = parser.parse_args()


def main():
    """Evaluate Inception-v3 on the test dataset."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Beginning evaluation using `birds_eval.test`.")
    birds_eval.test(vars(args))
    logging.info("Evaluation using `birds_eval.test` complete.")


if __name__ == "__main__":
    main()
