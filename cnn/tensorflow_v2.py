import os
import time
import numpy as np
import tensorflow as tf
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9

SPLIT_RATIO = [7, 1, 2] # [train, val, test]

def to_categorical(y, nb_classes):
  y = np.array(y)
  return (y[:, None] == np.unique(y)).astype(np.float32)

def variable_summary(var):
  # if not tf.get_variable_scope().reuse:
  #   name = var.op.namelogging.info('creating summary for: %s' % name)
  mean = tf.reduce_mean(var)
  tf.summary.scalar('mean', mean)
  with tf.name_scope('stddev'):
    stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
  tf.summary.scalar('stddev', stddev)
  tf.summary.scalar('max', tf.reduce_max(var))
  tf.summary.scalar('min', tf.reduce_min(var))
  tf.summary.histogram('histogram', var)


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
  # image = tf.clip_by_value(image, 0.0, 1.0)
  return image, label

def load_images(data_path):
  all_classes, all_images, all_labels = [], [], []
  for i in os.listdir(data_path):
    current_dir = os.path.join(data_path, i)
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
  all_labels = to_categorical(all_labels, 2)

  return all_images, all_labels, all_classes

def create_datasets(data_path, batch_size, resize, augment, num_threads, sizes=SPLIT_RATIO):
  all_images, all_labels, all_classes = load_images(data_path)
  total = all_images.shape[0]
  [train_size, val_size, test_size] = (sizes / np.sum(sizes) * total).tolist()
  train_size, val_size = np.rint([train_size, val_size]).astype(int).tolist()
  test = total - train_size - val_size
  
  print('Training size: %d' % train_size)
  print('Validation size: %d' % val_size)
  print('Test size: %d' % test)

  dataset = tf.data.Dataset.from_tensor_slices((all_images, all_labels))
  dataset = dataset.map(parse_dataset, num_parallel_calls=num_threads)
  dataset = dataset.shuffle(np.sum(sizes))

  if resize is not False:
    dataset = dataset.map(resize_dataset(resize), num_parallel_calls=num_threads)

  train_dset = dataset.take(train_size + val_size)
  test_dset = dataset.skip(train_size + val_size)
  val_dset = train_dset.skip(train_size)
  
  train_dset = train_dset.shuffle(train_size)
  val_dset = val_dset.shuffle(val_size)

  if augment:
    train_dset = train_dset.map(resize_dataset(resize), num_parallel_calls=num_threads)

  train_dset = train_dset.batch(batch_size).prefetch(1)
  val_dset = val_dset.batch(batch_size).prefetch(1)
  test_dset = test_dset.batch(batch_size)

  return train_dset, val_dset, test_dset, val_size, all_classes

def weights_variable(shape):
  return tf.get_variable('Weights', shape, initializer=tf.contrib.layers.xavier_initializer(seed=1))

def biases_variable(shape):
  return tf.get_variable('Biases', shape, initializer=tf.zeros_initializer())

def conv_layer(inputs, ksize, filters, layer_name='Conv', padding='SAME', stride=1, act=tf.nn.relu):
  with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as scope:
    in_channels = inputs.shape[-1]
    with tf.name_scope('Weights'):
      weights = weights_variable([ksize, ksize, in_channels, filters])
      variable_summary(weights)

    with tf.name_scope('Biases'):
      biases = biases_variable([filters])
      variable_summary(biases)

    with tf.name_scope('Linear_Compute'):
      preactivate = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding=padding) + biases
      tf.summary.histogram('linear', preactivate)

    activations = act(preactivate, name=scope.name)
    tf.summary.histogram('Activations', activations)

  return activations

def maxpool(inputs, ksize, stride, padding='VALID', layer_name='Maxpool'):
  with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as scope:
    pool = tf.nn.max_pool(inputs, ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1], padding=padding)
  return pool

def fully_connected(inputs, out_dim, layer_name='Fully_Connected'):
  with tf.variable_scope(layer_name, reuse=tf.AUTO_REUSE) as scope:
    in_dim = inputs.shape[-1]
    with tf.name_scope('Weights'):
      weights = weights_variable([in_dim, out_dim])
      variable_summary(weights)

    with tf.name_scope('Biases'):
      biases = biases_variable([out_dim])
      variable_summary(biases)

    with tf.name_scope('Linear_compute'):
      out = tf.matmul(inputs, weights) + biases
      tf.summary.histogram('Linear', out)

  return out

class LakeNet(object):
  def __init__(self, size, data_path='data', log_path='./logs', run_id='lakenet', learning_rate=0.0001, num_classes=2, batch_size=16, keep_prob=0.7):
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.keep_prob = tf.constant(keep_prob)
    self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='Global_Step')
    self.num_classes = num_classes
    self.skip_step = 5
    self.n_test = 100
    self.training = True
    self.size = size
    self.log_path = log_path
    self.run_id = run_id
    self.data_path = data_path

  def get_data(self):
    with tf.name_scope('Data'):
      train_dset, val_dset, test_dset, val_size, label_names = create_datasets(data_path=self.data_path, resize=self.size, augment=True, batch_size=self.batch_size, num_threads=16)
      # train_dset, val_dset = utils.get_mnist_dataset(self.batch_size)
      self.n_test = val_size
      iterator = tf.data.Iterator.from_structure(train_dset.output_types, train_dset.output_shapes)
      self.img, self.label = iterator.get_next()
      tf.summary.image('Image', self.img, 8)
      self.train_init = iterator.make_initializer(train_dset)  # initializer for train_data
      self.val_init = iterator.make_initializer(val_dset)    # initializer for val_data
      self.label_names = label_names

  def inference(self):
    net = conv_layer(inputs=self.img, filters=32, ksize=3, layer_name='Conv1')
    net = conv_layer(inputs=net, filters=32, ksize=3, layer_name='Conv2')
    net = maxpool(net, 2, 2, layer_name='Maxpool1')
    net = conv_layer(inputs=net, filters=64, ksize=3, layer_name='Conv3')
    net = conv_layer(inputs=net, filters=64, ksize=3, layer_name='Conv4')
    net = maxpool(net, 2, 2, layer_name='Maxpool2')
    net = conv_layer(inputs=net, filters=128, ksize=3, layer_name='Conv5')
    net = conv_layer(inputs=net, filters=128, ksize=3, layer_name='Conv6')
    net = maxpool(net, 2, 2, layer_name='Maxpool3')
    net = conv_layer(inputs=net, filters=512, ksize=3, layer_name='Conv7')
    net = conv_layer(inputs=net, filters=512, ksize=3, layer_name='Conv8')
    net = maxpool(net, 2, 2, layer_name='Maxpool4')

    net = tf.contrib.layers.flatten(net)
    net = fully_connected(net, 4096, layer_name='Fully_Connected1')
    net = tf.nn.relu(net, name='Relu1')

    net = tf.layers.dropout(net, self.keep_prob, training=self.training, name='Dropout1')

    net = fully_connected(net, 4096, layer_name='Fully_Connected2')
    net = tf.nn.relu(net, name='Relu2')

    net = tf.layers.dropout(net, self.keep_prob, training=self.training, name='Dropout2')

    self.logits = fully_connected(net, self.num_classes, layer_name='Fully_Connected3')

  def loss(self):
    with tf.name_scope('Loss'):
      entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.label, logits=self.logits)
      self.loss = tf.reduce_mean(entropy, name='Loss')

  def optimize(self):
    # learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 10000, 0.1, staircase=True, name='Decaying_Learning_Rate')
    # self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
    # tf.summary.scalar('Learning_Rate', learning_rate)
    self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step=self.global_step)

  def summary(self):
    with tf.name_scope('Performance'):
      tf.summary.scalar('Loss', self.loss)
      tf.summary.scalar('Accuracy', self.accuracy)
      tf.summary.histogram('Loss/Histogram', self.loss)
    self.summary_op = tf.summary.merge_all()

  def eval(self):
    with tf.name_scope('Predict'):
      preds = tf.nn.softmax(self.logits)
      correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
      self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    with tf.name_scope('Missclassification'):
      wrong_preds = tf.logical_not(correct_preds)
      wrong1 = tf.cast(tf.boolean_mask(tf.argmax(preds, 1), wrong_preds), dtype=tf.bool)
      wrong0 = tf.logical_not(wrong1)
      wrong0_img = tf.boolean_mask(tf.boolean_mask(self.img, wrong_preds), wrong0)
      wrong1_img = tf.boolean_mask(tf.boolean_mask(self.img, wrong_preds), wrong1)
      tf.summary.image('Misclassified_' + self.label_names[0], wrong0_img, 8)
      tf.summary.image('Misclassified_' + self.label_names[1], wrong1_img, 8)
  
  def build(self):
    with tf.device('/cpu:0'):
      self.get_data()
    with tf.device('/gpu:0'):
      self.inference()
      self.loss()
      self.optimize()
      self.eval()
      self.summary()

  def train_one_epoch(self, sess, saver, init, writer, epoch, step):
    start_time = time.time()
    sess.run(init)
    self.training = True
    total_loss = 0
    n_batches = 0

    try:
      while True:
        if step % 50 == 49:
          run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
          run_metadata = tf.RunMetadata()
          loss, summaries, _ = sess.run([self.loss, self.summary_op, self.opt], options=run_options, run_metadata=run_metadata)
          writer.add_run_metadata(run_metadata, 'step%d' % step)
          writer.add_summary(summaries, global_step=step)
          if (step + 1) % self.skip_step == 0:
            print('Loss at step {0}: {1}'.format(step, loss))
        else:
          loss, summaries, _ = sess.run([self.loss, self.summary_op, self.opt])
          writer.add_summary(summaries, global_step=step)
          if (step + 1) % self.skip_step == 0:
            print('Loss at step {0}: {1}'.format(step, loss))
        step += 1
        total_loss += loss
        n_batches += 1
    except tf.errors.OutOfRangeError:
      pass
    saver.save(sess, self.log_path + '/' + self.run_id + '.ckpt', step)
    print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
    print('Took: {0} seconds'.format(time.time() - start_time))

    return step

  def eval_once(self, sess, init, writer, epoch, step):
    start_time = time.time()
    sess.run(init)
    self.training = False
    total_correct_preds = 0
    try:
      while True:
        accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
        writer.add_summary(summaries, global_step=step)
        total_correct_preds += accuracy_batch
    except tf.errors.OutOfRangeError:
      pass

    print('Validation Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.n_test))
    print('Took: {0} seconds'.format(time.time() - start_time))


  def train(self, n_epochs):
    print('---------------------------------')
    print('Run id: ' + self.run_id)
    print('Log directory: ' + self.log_path + '/' + self.run_id)
    print('---------------------------------')
    tf.gfile.MakeDirs(self.log_path + '/' + self.run_id + '.ckpt')
    tf.set_random_seed(999)
    writer = tf.summary.FileWriter(self.log_path + '/' + self.run_id, tf.get_default_graph())

    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      saver = tf.train.Saver()
      ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.log_path + '/' + self.run_id + '.ckpt'))
      if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
      
      step = self.global_step.eval()

      for epoch in range(n_epochs):
        step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
        self.eval_once(sess, self.val_init, writer, epoch, step)
    writer.close()

if __name__ == '__main__':
  model = LakeNet(256, data_path='/mnt/hdd/luoqixin/lakes/')
  model.build()
  model.train(n_epochs=30)

