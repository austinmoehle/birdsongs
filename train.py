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
parser.add_argument('--model_path', default='checkpoints/inception_v3.ckpt',
                    help="Pretrained checkpoint for initialization.")
parser.add_argument('--restore', action='store_true',
                    help="Whether to restore from savefile in `restore_dir`.")
parser.add_argument('--restore_dir', default='runs/0/save/full',
                    help="Location of trained model to restore if `restore`=True.")
parser.add_argument('--batch_size', default=8, type=int,
                    help="Batch size.")
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers.")
parser.add_argument('--num_epochs1', default=20, type=int,
                    help="Number of epochs to train final layer (logits).")
parser.add_argument('--num_epochs2', default=20, type=int,
                    help="Number of epochs to train full net.")
parser.add_argument('--learning_rate1', default=1e-3, type=float,
                    help="Learning rate for training of final layer (logits).")
parser.add_argument('--learning_rate2', default=1e-5, type=float,
                    help="Learning rate for full net training.")
parser.add_argument('--epsilon', default=0.01, type=float,
                    help="Epsilon parameter for Adam optimizer.")
parser.add_argument('--dropout_keep_prob', default=0.8, type=float,
                    help="Dropout keep_prob.")
parser.add_argument('--weight_decay', default=4e-5, type=float,
                    help="Weight decay for weights variables.")
args = parser.parse_args()



def main():
    """Finetune Inception-v3 on the spectrogram dataset."""
    logging.basicConfig(level=logging.INFO)
    logging.info("Beginning training using `birds_train.finetune`")
    birds_train.finetune(vars(args))
    logging.info("Training using `birds_train.finetune` complete.")


if __name__ == "__main__":
    main()