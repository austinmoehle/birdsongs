"""
TensorFlow script for finetuning Inception-v3 on audio spectrogram data.

The code in model/finetune.py is based on an example by Olivier
Moindrot for finetuning the VGG model:

(https://gist.github.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/)

Many changes were made to suit the differences in model architecture
(Inception-v3 vs. VGG) and data type (spectrograms vs. standard images).

I also added a number of important features:

* model saver to save trained model weights
* loss monitoring, in addition to train/val accuracy
* summaries to visualize accuracy/loss as a function of epoch in TensorBoard
* ability to view sample images and associated labels during training, as a
  fun visualization and sanity check to confirm the correctness of class labels
* improved optimizer (Adam vs. simple gradient descent)
* class balancing (the birdsong dataset has an unbalanced class distribution)
* automatic logging of hyperparameters and results for hyperparameter search
* image visualization during evaluation plus random samples of true/predicted
  classes. This helps us make sense of the model's predictions, since the ID
  of a birdsong spectrogram isn't obvious to the average human observer.
"""

from __future__ import division
from __future__ import print_function

from model import birds_train

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
parser.add_argument('--run_dir', default='runs/0',
                    help="Directory to store training run.")
parser.add_argument('--init_path', default='checkpoints/inception_v3.ckpt',
                    help="Pretrained checkpoint for initialization.")
parser.add_argument('-i', '--initialize', action='store_true',
                    help="Whether to initialize from pretrained checkpoint.")
parser.add_argument('--restore_dir', default=None,
                    help="Directory to restore from, if `initialize` is False.")
parser.add_argument('--restore_path', default=None,
                    help="Path of saved model to restore.")
parser.add_argument('--batch_size', default=8, type=int,
                    help="Batch size.")
parser.add_argument('--num_workers', default=2, type=int,
                    help="Number of workers.")
parser.add_argument('--num_epochs', default=1, type=int,
                    help="Number of epochs to train.")
parser.add_argument('-f', '--freeze_conv_layers', action='store_true',
                    help="Whether to freeze convolutional layers during training.")
parser.add_argument('--learning_rate', default=1e-5, type=float,
                    help="Learning rate parameter for Adam optimizer.")
parser.add_argument('--epsilon', default=1e-8, type=float,
                    help="Epsilon parameter for Adam optimizer.")
parser.add_argument('--dropout_keep_prob', default=0.8, type=float,
                    help="Dropout keep_prob.")
parser.add_argument('--weight_decay', default=4e-5, type=float,
                    help="Weight decay for weights variables.")
args = parser.parse_args()



def main():
    """Finetune Inception-v3 on the spectrogram dataset."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Beginning training using `birds_train.finetune`...")
    birds_train.finetune(vars(args))
    logging.info("Training using `birds_train.finetune` complete.")


if __name__ == "__main__":
    main()

# Notes:
# approx time to run training
####LOGITS
# 10 epochs, acc every 5:  0:24:32
# 10 epochs, acc every 1:  0:41:38
# 30 epochs, acc every 10: 1:05:28

# With balancing, 200 epochs, acc every 20: 6:45:15


####FULL NET
# 25x2 epochs, acc every 20: 4:49:58 (10 e/hr)
