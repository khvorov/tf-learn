import numpy as np
import pickle
import os
import download

# Directory where you want to download and save the data-set.
# Set this before you start calling any of the functions below.
data_path = "data/CIFAR-10/"

# URL for the data-set on the internet.
data_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

img_size = 32
num_channels = 3

_num_files_train = 5
_batches_meta = None

def _get_file_path(filename):
    return os.path.join(data_path, 'cifar-10-batches-py', filename)

def _unpickle(filename):
    file_path = _get_file_path(filename)
    print('Loading data: ' + file_path)
    
    with open(file_path, mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
    
    return data

def _get_batches_meta():
    global _batches_meta
    if _batches_meta is None:
        _batches_meta = _unpickle('batches.meta')
    return _batches_meta

def _convert_images(raw):
    raw_float = np.array(raw, dtype=float) / 255.0
    return raw_float.reshape([-1, num_channels, img_size, img_size]).transpose([0, 2, 3, 1])

def _load_data(filename):
    data = _unpickle(filename)

    cls = np.array(data[b'labels'])
    images = _convert_images(data[b'data'])
    
    return images, cls

def load_class_names():
    raw = _get_batches_meta()[b'label_names']
    return [x.decode('utf-8') for x in raw]

def num_cases_per_batch():
    return _get_batches_meta()[b'num_cases_per_batch']

def load_training_data():
    num_images_train = _num_files_train * num_cases_per_batch()
    
    images = np.zeros(shape=[num_images_train, img_size, img_size, num_channels], dtype=float)
    cls = np.zeros(shape=[num_images_train], dtype=float)
    
    begin = 0
    
    for i in range(_num_files_train):
        images_batch, cls_batch = _load_data(filename='data_batch_' + str(i + 1))
        num_images = len(images_batch)
        end = begin + num_images
        images[begin:end, :] = images_batch
        cls[begin:end] = cls_batch
        begin = end
    
    return images, cls, None # TODO: one_hot_encoded

def load_test_data():
    images, cls = _load_data(filename='test_batch')
    return images, cls, None

def maybe_download_and_extract():
    download.maybe_download_and_extract(url=data_url, download_dir=data_path)

