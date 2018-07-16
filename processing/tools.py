from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
import pickle
import shutil
from subprocess import call
from shutil import copyfile
from scipy.io import wavfile
import scipy.misc
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import logging



def write_labels_file(labels_to_class_names, dataset_dir,
                      filename='labels.txt'):
    """Writes a file with the list of class names to 'dataset_dir/filename'.

    Args:
        labels_to_class_names: A map of (integer) labels to class names.
        dataset_dir: The directory in which the labels file should be written.
        filename: The filename where the class names are written.
    """
    labels_path = os.path.join(dataset_dir, filename)
    with open(labels_path, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def _make_bird_dirs(snip_dir, class_list):
    """Makes subdirectories in 'snip_dir' based on 'class_list'.

    Subdirectories are labeled numerically from 0 through len(class_list)
    in 'snip_dir'. If a subdirectory already exists, its contents are erased.

    Args:
        snip_dir: The destination directory
        class_list: A list of class names (238 birds).
    """
    for i in range(len(class_list)):
        path = os.path.join(snip_dir, str(i))
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)


def extract_description(path):
    """Returns the description from a filepath.

    Example:
        >>> print(extract_description('data/birds/hummingbird3.wav'))
        hummingbird3
    """
    return os.path.splitext(os.path.basename(path))[0]


def extract_bird_name(filename, split=None):
    """Returns the bird name from the filename of a Master Set recording.

    Example:
        >>> print(extract_bird_name('Acadian Flycatcher 2 MA Song.mp3'))
        Acadian Flycatcher
    """
    description = ''
    if split is None:
        words = filename.split()
    else:
        words = filename.split(split)
    for word in words:
        if is_number(word):
            break
        if description is '':
            description += word
        else:
            description += ' ' + word
    return description


def unpack_raw_master_set(directories, target, match='Song.mp3', dummy=True,
                           bird_list_path='bird_list.txt',
                           counts_path='counts.txt'):
    """Unpacks Master Set recordings in `directories` list.

    Copies files with names ending in 'match' from source directories to target
    directory. Counts the files per bird type and creates a list of
    bird names ordered by # of available recordings (descending).

    The Inception-v3 model requires an unused dummy class, which can be added
    to 'bird_list' by specifying dummy=True.

    Example w/ match='Song.mp3':
        'Acadian Flycatcher 2 MA Song.mp3'            --> included
        'Carolina Chickadee 9 MD Chickadee calls.mp3' --> not included

    Example file written to 'bird_list_path':
        dummy
        Tufted Titmouse
        White-throated Sparrow
        Tennessee Warbler
        Magnolia Warbler
        White-winged Crossbill
        .
        .
        .

    Example file written to 'counts_path':
        Tufted Titmouse: 22
        White-throated Sparrow: 19
        Tennessee Warbler: 15
        Magnolia Warbler: 15
        White-winged Crossbill: 13
        .
        .
        .
    """
    if not isinstance(directories, list):
        directories = [directories]
    bird_counts = {}
    for directory in directories:
        for filename in os.listdir(directory):
            words = filename.split()
            if words[-1] == match:
                bird_name = extract_bird_name(filename)
                bird_counts[bird_name] = bird_counts.get(bird_name, 0) + 1
                copyfile(os.path.join(directory, filename),
                         os.path.join(target, filename))
    bird_list = sorted(bird_counts.keys(),
                       key=lambda x: bird_counts[x], reverse=True)
    if dummy:
        bird_list.insert(0, dummy)
    with open(bird_list_path, 'w') as f:
       for bird in bird_list:
           f.write('%s\n' % bird)
    with open(counts_path, 'w') as f:
        for bird, count in reversed(sorted(bird_counts.iteritems(),
                                    key=lambda k_v: (k_v[1], k_v[0]))):
            f.write('%s: %d\n' % (bird, count))


def mp3_to_wav(song_dir, snip_dir, bird_list_path='bird_list.txt'):
    """Converts .mp3 files in 'song_dir' to .wav and move them to 'snip_dir'.

    For each bird, the .wav files are placed in a subdirectory in 'snip_dir'
    named using that bird's numerical label, as provided by bird_list.

    This function requires the ffmpeg package to convert from .mp3 to .wav.
    """
    if os.path.exists(snip_dir):
        shutil.rmtree(snip_dir)
    os.makedirs(snip_dir)
    with open(bird_list_path) as f:
        lines = f.readlines()
    bird_list = [line.rstrip('\n') for line in lines]
    # Build the bird-labeled subdirectories in 'snip_dir'.
    _make_bird_dirs(snip_dir, birds_list)
    # Populate the subdirectory with recordings converted from .mp3 to .wav.
    for f in os.listdir(song_dir):
        bird = extract_bird_name(f)
        if bird in birds_list:
            index = birds_list.index(bird)
            wav_filename = os.path.splitext(f)[0].replace(' ', '_') + '.wav'
            orig = os.path.join(mp3_dir, f)
            new = os.path.join(snip_dir, str(index), wav_filename)
            # MP3-to-WAV conversion requires the ffmpeg package.
            call(["ffmpeg", "-i", orig, new])


def trim_audio(data, rate=44100, start_trim=0, end_trim=0, log=False):
    """Trims seconds from the start and end of an audio file.

    Parameters 'start_trim' and 'end_trim' given in seconds.
    """
    chop = np.copy(data[start_trim*rate : len(data)-end_trim*rate])
    if log:
        m, s = divmod(float(len(data))/rate, 60)
        h, m = divmod(m, 60)
        logging.info("Original recording length: %d h %d m %d s" % (h, m, s))
        logging.info("Removed [%d s, %d s] from [start, end] of recording." %
              (start_trim, end_trim))
    return chop


def find_loudest_subset(data, subset_length, step=0.1, rate=44100):
    """Finds the loudest subset in an array.

    Note: this uses a weighted window function ('abs_weights'), which makes it
          more computationally expensive than finding a simple array subset.

    Params 'subset_length' and 'step' (search step size) given in seconds.
    """
    rate = float(rate)
    step_samples = int(step * rate)
    subset_samples = int(subset_length * rate)
    wav_scale = 2**15
    data_squared = (data / float(wav_scale)) ** 2
    max_sum = 0
    start_idx = 0
    end_idx = subset_samples
    weights = welch_weights(subset_samples)
    for i in range(int((len(data) - subset_samples + 1.0)/step_samples)):
        idx = i * step_samples
        if weighted_sum(data_squared[idx : idx+subset_samples], weights) > max_sum:
            max_sum = weighted_sum(data_squared[idx : idx+subset_samples], weights)
            start_idx = idx
            end_idx = idx + subset_samples
    return max_sum, start_idx, end_idx


def welch_weights(length):
    center = (length - 1) / 2.0
    weights = np.zeros(length)
    for i in xrange(length):
        weights[i] = 1.0 - ((i-center)/center)**2
    return weights


def abs_weights(length):
    center = (length - 1) / 2.0
    weights = np.zeros(length)
    for i in xrange(length):
        weights[i] = 1.0 - abs((i-center)/center)
    return weights


def weighted_sum(array, weights):
    return sum(weights * array)


def snip_audio(data, snip_length=4, cutoff=0.25, min_snips=None, max_snips=None,
               num_jitters=None, jitter=0.25,
               rate=44100, log=False):
    """Pads the raw audio, then snips the "loudest" snippets of length
    'snip_length' (in seconds). Saves the "volume" (weighted sum of samples)
    of the "loudest" snippet as 'max_sum'.

    When 'min_snips' is None, this function continues to extract snippets until
    the snippet "volume" (total sum of audio data over the subset w/ weighting
    window applied) falls below cutoff*max_sum.

    Can apply a random time jitter ('num_jitters' @ 'jitter' seconds)
    for data augmentation.

    Returns:
        snips: Spliced data.
        logs:  Logs containing information about the splice indices and max_sum.
    """
    if max_snips is None:
        if min_snips is None:
            min_snips = 1
        max_snips = max(min_snips, int((float(len(data))/rate)/3.0))
    # Pad data with (snip_length * rate / 2) zeros.
    chop = np.lib.pad(data, int(snip_length*rate/2), 'constant')
    if log:
        logging.info("Data padded with %.1f s of zeros." %
                     (float(snip_length)/2))
    snips = []
    logs = []
    max_sum = 0
    count = 0
    sum_ratio = 1

    while True:
        current_sum, start_idx, end_idx = find_loudest_subset(chop, snip_length,
                                                              rate=rate)
        max_sum = max(max_sum, current_sum)
        sum_ratio = float(current_sum) / max_sum
        if sum_ratio < cutoff:
            break
        collection = []
        if num_jitters is None:
            collection.append(np.copy(chop[start_idx : end_idx]))
        else:
            for j in xrange(num_jitters):
                offset = int(jitter * rate * random.uniform(-1, 1))
                try:
                    collection.append(np.copy(chop[start_idx+offset : end_idx+offset]))
                except IndexError:
                    collection.append(np.copy(chop[start_idx : end_idx]))
        logs.append((sum_ratio, max_sum, start_idx, end_idx))
        chop[start_idx : end_idx] = 0
        snips.append(collection)
        count += 1
        if count >= max_snips:
            break
    return snips, logs


def plot_spectrograms(data, rate=44100, num_figures=1, fig_size=(16,9),
                      sample_index=0, n_mels=256):
    # Produces spectrograms with varying window and hop lengths.
    fig = plt.figure(figsize=(16,9))
    if type(data) is not list:
        num_figures = 1
    shapes = np.zeros((num_figures, 2))
    for i in xrange(num_figures):
        if sample_index is None:
            y = data[i] if type(data) is list else data
            hop_length = 512
            n_fft = 2048
        else:
            y = data[sample_index] if type(data) is list else data
            hop_length = int(512 / (2**i))
            n_fft = int(2048 / (2**i))
        S = librosa.feature.melspectrogram(y, sr=rate, hop_length=hop_length,
                                           n_fft=n_fft, n_mels=n_mels)
        log_S = librosa.logamplitude(S, ref_power=np.max)
        shapes[i] = (log_S.shape[1], log_S.shape[0])
        fig.add_subplot(2,2,i+1)
        librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
        plt.title(bird_name + ' #' + str(i) + ' ' + str(n_fft) + '/' + str(hop_length))
        plt.colorbar(format='%+02.0f dB')
        plt.tight_layout()
    plt.show()
    logging.info('[Time subdivisions, frequency subdivisions]:')
    logging.info(shapes)


def compute_spectrograms(data, rate=44100, fig_size=(16,9), sample_index=None,
                         n_mels=256, n_fft=2048, hop_length=512):
    """Computes spectrograms from audio data."""
    spectrograms = []
    n_ffts = []
    i = 0
    if sample_index is not None or type(data) is not list:
        y = data[sample_index] if type(data) is list else data
        hop = int(hop_length / (2**i))
        n = int(n_fft / (2**i))
    else:
        y = data[i] if type(data) is list else data
        hop = hop_length
        n = n_fft
    try:
        S = librosa.feature.melspectrogram(y, sr=rate, hop_length=hop,
                                           n_fft=n, n_mels=n_mels)
    except IndexError:
        logging.info('Spectrogram index error. Skipping this spectrogram.')
        return
    log_S = librosa.logamplitude(S, ref_power=np.max)
    spectrograms.append(log_S)
    n_ffts.append(n)
    return spectrograms, n_ffts


def spectrogram_to_jpg(image, description, target='./'):
    path = os.path.join(target, description + '.jpg')
    fig = librosa.display.specshow(image, sr=44100, x_axis='time', y_axis='mel',
                                   cmap='gray_r')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(path, bbox_inches='tight', pad_inches = 0)
    return path


def raw_audio_to_jpgs(data, target, sample_dir, bird_index, rec_index,
                      cutoff=0.25, min_snips=None,
                      snip_length=4, rate=44100, num_jitters=None, jitter=0.25):
    """Generates spectrograms from a full audio recording (.wav).

    The spectrograms are placed in the target folder. 'target' should be either
    'train' or 'test'.

    One spectrogram from each recording is added to 'sample_dir' for later
    viewing and evaluation.
    """
    snippets, logs = snip_audio(data, snip_length=snip_length, cutoff=cutoff,
                                min_snips=min_snips, num_jitters=num_jitters,
                                jitter=jitter, rate=rate)
    first = True
    for i, collection in enumerate(snippets):
        for j, snip in enumerate(collection):
            if first:
                # raw_to_wav then write to file in 'sample_dir'
                raw_to_wav(snip, os.path.join(sample_dir, str(rec_index) + '.wav'))
                first = False
            spectrograms = compute_spectrograms(snip)
            for k in range(len(spectrograms[0])):
                start_time = float(logs[i][2])/rate
                label = str(bird_index) + '_' + str(rec_index)
                label += '_i%d_%dp%d_c%d' % (i, int(start_time),
                                             int((start_time % 1)*10),
                                             int(100.*logs[i][0]))
                path = spectrogram_to_jpg(spectrograms[0][k], label, target=target)


def wav_to_raw(path, log=False):
    """Converts WAV file to raw audio data."""
    rate, data = wavfile.read(path)
    if log:
        m, s = divmod(float(len(data))/rate, 60)
        h, m = divmod(m, 60)
        logging.info("Original recording length: %d h %d m %d s" % (h, m, s))
    try:
        if data.shape[1] == 2:
            # If stereo (2-channel), take the average of the two channels.
            data = 0.5 * (data[:, 0] + data[:, 1])
            if log:
                logging.info('Stereo audio')
    except IndexError:
        if log:
            logging.info('Mono audio')
    return rate, data


def raw_to_wav(data, path, rate=44100):
    """Writes raw audio chunk 'data' to 'path' in WAV format."""
    wavfile.write(path, rate, data)


def add_snippets_from_file(path, target, sample_dir, bird_index, rec_index,
                           cutoff=0.25, min_snips=None, snip_length=4,
                           num_jitters=None, jitter=0.25):
    """Adds snippets from an audio file."""
    rate, data = wav_to_raw(path)
    if rate != 44100:
        logging.info('Rate is not 44100 Hz (%s Hz)' % str(rate))
    raw_audio_to_jpgs(data, target, sample_dir, bird_index, rec_index,
                      cutoff=cutoff,
                      min_snips=min_snips,
                      snip_length=snip_length,
                      rate=rate,
                      num_jitters=num_jitters,
                      jitter=jitter)


def make_snippets(snip_dir, snip_length=4, num_jitters=10, jitter=0.5):
    """Make spectrogram snippets from full recordings in `snip_dir`.

    Searches each class directory in 'snip_dir' for audio files. For each
    file found, create a number-labeled subdirectory in the class directory and
    fill it with spectrograms snipped from that recording.

    Note: it may take a long time to generate the spectrograms. A progress list
    is created at `snip_completion.pkl` so that we can interrupt and resume from
    the checkpoint.

    Args:
    snip_dir:    The parent directory containing class-labeled subdirectories
                 containing the recordings.
    snip_length: Desired snippet length in seconds (default 4).
    num_jitters: (int) Number of "duplicates" of each snip to create for data-
                 augmentation purposes. Each "duplicate" has a jitter applied in
                 the time-dimension to slightly randomize the image created.
    jitter:      (float) The amount of jitter to be applied, in seconds.
    """
    # Load progress list if it exists.
    if os.path.isfile('snip_completion.pkl'):
        with open('snip_completion.pkl', 'rb') as fp:
            completion_list = pickle.load(fp)
    # Loop through all birds in the dataset.
    for subd in os.listdir(snip_dir):
        if int(subd) in completion_list or int(subd) > 48:
            continue
        bird_path = os.path.join(snip_dir, subd)
        if not os.path.isdir(bird_path):
            continue
        sample_dir = os.path.join(bird_path, 'samples')
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)
        os.makedirs(sample_dir)
        index = int(subd)
        curr = 0
        # Loop through all recordings for this bird.
        for rec in sorted(os.listdir(bird_path)):
            rec_file = os.path.join(bird_path, rec)
            if not os.path.isfile(rec_file):
                continue
            rec_dir = os.path.join(bird_path, str(curr))
            if os.path.exists(rec_dir):
                shutil.rmtree(rec_dir)
            os.makedirs(rec_dir)
            # Make spectrogram snippets from this particular audio file.
            add_snippets_from_file(rec_file, rec_dir, sample_dir, index, curr,
                                   snip_length=snip_length,
                                   num_jitters=num_jitters,
                                   jitter=jitter)
            curr += 1
        logging.info('Finished with %d recordings for bird %d' % (curr, index))
        # Add this bird label to the completion list.
        completion_list.append(int(subd))
        with open('snip_completion.pkl', 'wb') as fp:
            pickle.dump(completion_list, fp)


def clear_directory(directory):
    [os.remove(f) for f in os.listdir(directory)]


def add_to_split(rec_dir, target, label):
    """Add snips from a recording to the target split. (Place in numbered directory according to 'label')."""
    for file_name in os.listdir(rec_dir):
        path = os.path.join(rec_dir, file_name)
        if (os.path.isfile(path)):
            if not os.path.isdir(os.path.join(target, str(label))):
                os.makedirs(os.path.join(target, str(label)))
            shutil.copy(path, os.path.join(target, str(label), file_name))


def add_to_split_numbered(rec_dir, target, label):
    """Add snips from a recording to the target split. (Place in numbered directory according to 'label')."""
    for file_name in os.listdir(rec_dir):
        path = os.path.join(rec_dir, file_name)
        if (os.path.isfile(path)):
            count = 0
            if os.path.isdir(os.path.join(target, str(label))):
                count = len([f for f in os.listdir(os.path.join(target, str(label)))])
            else:
                os.makedirs(os.path.join(target, str(label)))
            shutil.copy(path, os.path.join(target, str(label), str(count)))


def snips_to_split(snip_dir, data_dir, test_recs=2, min_recs=6,
                   bird_list_path='bird_list.txt',
                   labels_file_path='labels.txt'):
    """Divides spectrograms in 'snip_dir' into train/test splits.

    Train/test splits are created at 'data_dir/train' and 'data_dir/test'.
    Each recording in 'snip_dir' is assigned to either the train or test split,
    which means that two spectrograms generated from the same recording will
    always end up in the same split.

    Also writes a labels file to 'data_dir', which is needed during training.

    Args:
        snip_dir:  Source directory (subdirectories contain spectrograms).
        data_dir:  Target directory.
        test_recs: Number of recordings from each bird to send to test split.
        min_recs:  Minimum number of recordings per bird. If there aren't
                   enough recordings, don't include this bird in the dataset.
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    os.makedirs(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir)

    for subd in os.listdir(snip_dir):
        bird_path = os.path.join(snip_dir, subd)
        if not os.path.isdir(bird_path):
            continue
        rec_folders = [sd for sd in os.listdir(bird_path) if
                       (os.path.isdir(os.path.join(bird_path, sd))
                       and sd != 'samples')]
        num_recordings = len(rec_folders)
        if num_recordings < min_recs:
            continue
        counts = [len([f for f in os.listdir(os.path.join(bird_path, sd))])
                  for sd in rec_folders]
        remaining = num_recordings
        while remaining > 0:
            max_index = counts.index(max(counts))
            # Send all recordings except test_recs=2 to train split.
            if remaining > test_recs:
                add_to_split(os.path.join(bird_path, rec_folders[max_index]),
                train_dir, int(subd))
            # Send remaining recordings to test split.
            else:
                add_to_split(os.path.join(bird_path, rec_folders[max_index]),
                test_dir, int(subd))
            counts[max_index] = 0
            remaining -= 1

    # Calculate and print the number of train and test images.
    subs = [sd for sd in os.listdir(train_dir) if
            os.path.isdir(os.path.join(train_dir, sd))]
    num = sum([len([f for f in os.listdir(os.path.join(train_dir, sd))])
              for sd in subs])
    logging.info("Number of images in train split: %d" % num)
    subs = [sd for sd in os.listdir(test_dir) if
            os.path.isdir(os.path.join(test_dir, sd))]
    num = sum([len([f for f in os.listdir(os.path.join(test_dir, sd))])
              for sd in subs])
    logging.info("Number of images in test split: %d" % num)

    # Get 'bird_list' and use it to write the labels file for training.
    with open(bird_list_path) as f:
        lines = f.readlines()
    bird_list = [line.rstrip('\n') for line in lines]
    labels_to_class_names = {label: bird for label, bird in
                             enumerate(bird_list) if label <= cutoff}
    write_labels_file(labels_to_class_names, data_dir,
                      filename=labels_file_path)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_counts_file(path):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
        dataset_dir: The directory in which the labels file is found.
        filename: The filename where the class names are written.

    Returns:
        A map from a label (integer) to class name.
    """
    labels_filename = path
    with open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_counts = {}
    for line in lines:
        index = line.index(':')
        labels_to_counts[line[:index]] = int(line[index+1:])
    return labels_to_counts
