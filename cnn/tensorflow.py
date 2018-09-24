from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', 'data', 'training data dir')
flags.DEFINE_string('log_path', 'logs', 'the log dir')
flags.DEFINE_float('learning_rate', 5e-7, 'learning rate')
flags.DEFINE_integer('num_classes', 2, 'number of classes')
flags.DEFINE_integer('batch_size', 8, 'size of a batch')
flags.DEFINE_integer('resize', 512, 'size to resize image')
flags.DEFINE_integer('num_epochs', 10, 'steps of epochs')
flags.DEFINE_integer('point_every', 10, 'display a period')
flags.DEFINE_bool('use_gpu', False, 'use gpu mode')
flags.DEFINE_bool('augment', True, 'augment samples')
flags.DEFINE_integer('num_threads', 8, 'number of threads')
flags.DEFINE_float('dropout', 0.9, 'percent of keep')

SPLIT_RATIO = [7, 1, 2] # [train, val, test]

if FLAGS.use_gpu:
    device = '/device:GPU:0'
else:
    device = '/cpu:0'


print('Using device: ', device)

def load_images(data_dir):
    all_classes, all_images, all_labels = [], [], []
    for i in os.listdir(data_dir):
        current_dir = os.path.join(data_dir, i)
        if os.path.isdir(current_dir):
            all_classes.append(i)
            for img in os.listdir(current_dir):
                if img.endswith('png'):
                    all_images.append(os.path.join(current_dir, img))
                    all_labels.append(all_classes.index(i))

    temp = np.array([all_images, all_labels]).T
    np.random.shuffle(temp)

    all_images = temp[:, 0]
    all_labels = temp[:, 1].astype(int)

    return all_images, all_labels

def parse_dataset(filename, label):
  image_string = tf.read_file(filename)
  image = tf.image.decode_png(image_string, channels=3)
  # convert to float values in [0, 1]
  # image = tf.image.convert_image_dtype(image, tf.float32)
  return image, label

def resize_dataset(size):
  def _resize_dataset(image, label):
    image = tf.image.resize_images(image, [size, size])
    return image, label
  return _resize_dataset

def augment_dataset(image, label):
  seed = random.randint(0, 2 ** 31 - 1)
  image = tf.image.random_flip_left_right(image, seed=seed)
  image = tf.image.random_flip_up_down(image, seed=seed)
  # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
  # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

  # make sure the image is still in [0, 1]
  # image = tf.clip_by_value(image, 0.0, 1.0)
  # image = tf.image.per_image_standardization(image)

  return image, label

def create_datasets(data_dir, batch_size, resize, augment, epochs, num_threads, sizes=SPLIT_RATIO):
  all_images, all_labels = load_images(data_dir)
  total = all_images.shape[0]
  [train_size, val_size, test_size] = (sizes / np.sum(sizes) * total).tolist()
  train_size, val_size = np.rint([train_size, val_size]).astype(int).tolist()
  test = total - train_size - val_size
  
  print(train_size, val_size, test)

  dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
  dataset = dataset.map(parse_dataset, num_parallel_calls=num_threads)
  dataset = dataset.shuffle(np.sum(sizes))

  if resize is not False:
    dataset = dataset.map(resize_dataset(resize), num_parallel_calls=num_threads)

  train_dset = dataset.take(train_size + val_size)
  test_dset = dataset.skip(train_size + val_size)
  val_dset = train_dset.skip(train_size)
  
  train_dset = train_dset.shuffle(train_size).repeat(epochs)
  val_dset = val_dset.shuffle(val_size)

  if augment:
    train_dset = train_dset.map(resize_dataset(resize), num_parallel_calls=num_threads)

  train_dset = train_dset.batch(batch_size).prefetch(1)
  val_dset = val_dset.batch(batch_size).prefetch(1)
  test_dset = test_dset.batch(batch_size)

  return train_dset, val_dset, test_dset

def activation_summary(activations):
  """Helper to create summaries for activations. Creates a summary that provides a histogram of activations. Creates a summary that measure the sparsity of activations. Parameters: ----------- activation : tensor, activation of layer. Returns: -------- no return """
  tensor_name = activations.op.name
  # tf.summary
  tf.summary.histogram(tensor_name + '/histogram', activations)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(activations))


def weight_summary(weights, bias):
  """Helper to create summaries for weights and bias. Creates a summary that provides a histogram of weights and bias. Parameters: ----------- weights : weights of layers. bias : bias of layers. Returns: -------- no return """
  op_name_weights = weights.op.name
  op_name_bias = bias.op.name
  tf.summary.histogram(op_name_weights + '/histogram', weights)
  tf.summary.histogram(op_name_bias + '/histogram', bias)

def model_init_fn(inputs, num_classes, is_training):
    channel_1, channel_2, = 16, 32
    initializer = tf.variance_scaling_initializer(scale=2.0)
    layers = [
        tf.layers.Conv2D(channel_1,kernel_size=5, strides=1, padding='SAME', activation=tf.nn.leaky_relu,
                            use_bias='TRUE', bias_initializer=initializer, kernel_initializer=initializer),
        tf.layers.Conv2D(channel_2,kernel_size=3, strides=1, padding='SAME', activation=tf.nn.leaky_relu,
                            use_bias='TRUE', bias_initializer=initializer, kernel_initializer=initializer),
        tf.layers.Flatten(),
        tf.layers.Dense(num_classes, use_bias='TRUE', bias_initializer=initializer,
                         kernel_initializer=initializer)
    ]
    model = tf.keras.Sequential(layers)
    return model(inputs)

def optimizer_init_fn():
    learning_rate = 5e-7
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9, use_nesterov=True)
    return optimizer

def check_accuracy(sess, handle, val_handle, val, x, scores, is_training=None):
    num_correct, num_samples = 0, 0
    try:
        while True:
            x_batch, y_batch = sess.run(val, feed_dict={ handle: val_handle })
            feed_dict = { x: x_batch, is_training: False }
            scores_np = sess.run(scores, feed_dict=feed_dict)
            y_pred = scores_np.argmax(axis=1)
            num_samples += x_batch.shape[0]
            num_correct += (y_pred == y_batch).sum()
    except tf.errors.OutOfRangeError:
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f%%)' % (num_correct, num_samples, 100 * acc))

def train(model_init_fn, optimizer_init_fn, num_epochs, batch_size, num_threads,
          learning_rate, size, point_every, data_path, augment, num_classes, log_path):
  tf.reset_default_graph()

  print(learning_rate, num_classes)

  with tf.device('/cpu:0'):

    train_dset, val_dset, test_dset = create_datasets(data_path, num_threads=num_threads, resize=size, augment=augment, batch_size=batch_size, epochs=num_epochs)
    
    train_iter = train_dset.make_initializable_iterator()
    val_iter = val_dset.make_initializable_iterator()
    test_iter = test_dset.make_initializable_iterator()

    handle = tf.placeholder(tf.string, shape=[])
    iter = tf.data.Iterator.from_string_handle(handle, train_dset.output_types, train_dset.output_shapes)
    elements = iter.get_next()


  with tf.device(device):

    with tf.name_scope('input'):
      x = tf.placeholder(tf.float32, [None, size, size, 3], name='input_image')
      y = tf.placeholder(tf.int32, [None], name='input_label')
    tf.summary.image('image', tf.cast(x, tf.uint8), batch_size * 2)

    is_training = tf.placeholder(tf.float32, name='is_training')

    # training accuracy
    scores = model_init_fn(x, num_classes, is_training)
    correct_prediction = tf.equal(tf.argmax(scores, 1), tf.argmax(y))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('training_accuracy', accuracy)

    # loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=scores)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('loss', loss)

    optimizer = optimizer_init_fn()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = optimizer.minimize(loss)

    summary = tf.summary.merge_all()

  with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(log_path, sess.graph)

    sess.run(tf.global_variables_initializer())
    
    train_handle = sess.run(train_iter.string_handle())
    val_handle = sess.run(val_iter.string_handle())
    test_handle = sess.run(test_iter.string_handle())

    sess.run(train_iter.initializer)
    sess.run(test_iter.initializer)

    t = 0
    try:
      while True:
        x_np, y_np = sess.run(elements, feed_dict={ handle: train_handle })
        feed_dict = { x: x_np, y: y_np, is_training: True }
        summary_np, loss_np, _ = sess.run([summary, loss, train_op], feed_dict=feed_dict)
        summary_writer.add_summary(summary_np, t)

        # if t % batch_size == 0:
          # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          # run_metadata = tf.RunMetadata()
          # summary_np = sess.run(summary, options=run_options, run_metadata=run_metadata)

          # summary_writer.add_run_metadata(run_metadata, 'step%03d' % t)
          # summary_writer.add_summary(summary_np, t)


        if t % point_every == 0:
          sess.run(val_iter.initializer)
          print('Iteration %d, loss = %.4f' % (t, loss_np))
          check_accuracy(sess, handle, val_handle, elements, x, scores, is_training=is_training)
        t += 1

    except tf.errors.OutOfRangeError:
      print('training ended')
      check_accuracy(sess, handle, test_handle, elements, x, scores, is_training=is_training)

    summary_writer.close()

train(model_init_fn, optimizer_init_fn, batch_size=FLAGS.batch_size, 
        point_every=FLAGS.point_every, augment=FLAGS.augment, log_path=FLAGS.log_path,
        num_epochs=FLAGS.num_epochs, learning_rate=FLAGS.learning_rate, num_threads=FLAGS.num_threads,
        size=FLAGS.resize, data_path=FLAGS.data_path, num_classes=FLAGS.num_classes)