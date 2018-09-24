import tflearn
import h5py
import os
import tensorflow as tf
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.data_utils import build_hdf5_image_dataset

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# build_hdf5_image_dataset('data', image_shape=(512, 512), output_path='dataset.h5',
                        # mode='folder', categorical_labels=True)


h5f = h5py.File('dataset512.h5', 'r')
X = h5f['X']
Y = h5f['Y']

X, Y = shuffle(X, Y)

img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_flip_updown()
img_aug.add_random_rotation(max_angle=25.)

network = input_data(shape=[None, 512, 512, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 64, 3, activation='relu')
network = conv_2d(network, 64, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = conv_2d(network, 128, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='leaky_relu', regularizer='L2')
network = dropout(network, 0.5)
network = fully_connected(network, 512, activation='leaky_relu', regularizer='L2')
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='softmax_categorical_crossentropy',
                     learning_rate=0.0001)


model = tflearn.DNN(network, tensorboard_verbose=3, tensorboard_dir='./logs')
model.fit(X, Y, n_epoch=200, shuffle=True, validation_set=0.1, show_metric=True, batch_size=8, run_id='lakenet_v2')