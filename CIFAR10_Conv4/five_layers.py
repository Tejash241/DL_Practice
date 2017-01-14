import cPickle as pickle
import numpy as np
import tensorflow as tf
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
tf.logging.set_verbosity(tf.logging.ERROR)
# =============================== UTILITY FUNCTIONS ==========================================
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = np.array(datadict['labels']).astype(np.int32)
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
    return X, Y

def load_CIFAR10(ROOT):
  """ load all of cifar """
  xs = []
  ys = []
  for b in range(1,6):
    f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
    X, Y = load_CIFAR_batch(f)
    xs.append(X)
    ys.append(Y)    
  Xtr = np.concatenate(xs)
  Ytr = np.concatenate(ys)
  del X, Y
  Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
  return Xtr, Ytr, Xte, Yte

def one_hot_vector(y_in, num_classes):
  y_out = np.zeros((y_in.shape[0], num_classes))
  y_out[np.arange(y_in.shape[0]), y_in] = 1
  return y_out

# =============================== NETWORK ============================================================
def init_variables(num_classes):
  weights = {
    'wc1':weight_variable([3, 3, 3, 32], 'wc1'),
    'wc2':weight_variable([5, 5, 32, 128], 'wc2'),
    'wc3':weight_variable([5, 5, 128, 256], 'wc3'),
    'wd1':weight_variable([8*8*256, 4096], 'wd1'),
    'wd2':weight_variable([4096, 4096], 'wd2'),
    'wout':weight_variable([4096, num_classes], 'wout')
  }
  biases = {
    'bc1':tf.Variable(tf.zeros([32]), name='bc1'),
    'bc2':tf.Variable(tf.zeros([128]), name='bc2'),
    'bc3':tf.Variable(tf.zeros([256]), name='bc3'),
    'bd1':tf.Variable(tf.zeros([4096]), name='bd1'),
    'bd2':tf.Variable(tf.zeros([4096]), name='bd2'),
    'bout':tf.Variable(tf.zeros([num_classes]), name='bout')
  }
  return weights, biases

def weight_variable(shape, name):
  # xavier initialization
  num = 6.0
  if len(shape) == 4:
      # xavier weights
      den = shape[0] * shape[1] * (shape[2] + shape[3])
  else:
      den = sum(shape)
  init_range = np.sqrt(num / den)
  return tf.Variable(tf.random_uniform(shape, minval=-init_range, maxval=init_range), name=name)

def conv2d(img, W, b, name, stride=1, padding='SAME', depth_radius=3):
  with tf.name_scope(name) as scope:
    network = tf.nn.bias_add(tf.nn.conv2d(img, W, strides=[1,stride,stride,1], padding=padding), b)
    out = tf.nn.relu(tf.nn.local_response_normalization(network, depth_radius)) 
  return out

def maxpool2d(img, k=1, padding='SAME'):
  return tf.nn.max_pool(img, ksize=[1,k,k,1], strides=[1,k,k,1], padding=padding)

def build(X, weights, biases, dropout=1, batch_size=32):
  conv1 = conv2d(X, weights['wc1'], biases['bc1'], 'conv1')
  # conv1 = maxpool2d(conv1, k=2)
  # conv1 = tf.nn.dropout(conv1, dropout)
  print 'CONV1', conv1.get_shape()
  conv2 = conv2d(conv1, weights['wc2'], biases['bc2'], 'conv2')
  conv2 = maxpool2d(conv2, k=2)
  # conv2 = tf.nn.dropout(conv2, dropout)
  print 'CONV2', conv2.get_shape()
  conv3 = conv2d(conv2, weights['wc3'], biases['bc3'], 'conv3')
  conv3 = maxpool2d(conv3, k=2)
  # conv3 = tf.nn.dropout(conv3, dropout)
  print 'CONV3', conv3.get_shape()
  shp = conv3.get_shape()
  new_shape = shp[1].value*shp[2].value*shp[3].value
  conv3 = tf.reshape(conv3, [-1, new_shape])
  print 'CONV4 reshaped', conv3.get_shape()
  fc1 = tf.nn.relu(tf.add(tf.matmul(conv3, weights['wd1']), biases['bd1']), name='fc1')
  fc1 = tf.nn.dropout(fc1, dropout)
  print 'FC1', fc1.get_shape()
  fc2 = tf.nn.relu(tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2']), name='fc2')
  fc2 = tf.nn.dropout(fc2, dropout)
  print 'FC2', fc2.get_shape()
  out = tf.nn.relu(tf.add(tf.matmul(fc2, weights['wout']), biases['bout']), name='predictions')
  print 'OUT', out.get_shape()
  return out


num_classes = 10
num_epochs = 50
batch_size = 128
height = 32
width = 32
num_channels = 3
with tf.Graph().as_default():
  X_ = tf.placeholder(tf.float32, [None, height, width, num_channels])
  y_ = tf.placeholder(tf.int32, [None, num_classes])
  keep_prob = tf.placeholder(tf.float32)          
  # correct_pred = tf.placeholder(tf.float32, [None])
  weights, biases = init_variables(num_classes)

  predictions = build(X_, weights, biases, keep_prob)
  entropy = tf.nn.softmax_cross_entropy_with_logits(predictions, y_, name='entropy')
  loss = tf.reduce_mean(entropy, name='loss')
  correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y_, 1), name='correct_pred')
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
  # optimizer = get_optimizer(loss)
  optimizer = tf.train.AdamOptimizer(learning_rate=.0001).minimize(loss)

  init = tf.global_variables_initializer()

  # ================================ TRAINING ======================================================
  with tf.Session() as sess:
    sess.run(init)
    writer = tf.train.SummaryWriter("train_logs", graph_def=sess.graph_def)
    tf.train.write_graph(sess.graph_def, './train_logs', 'train.pbtxt')
    saver = tf.train.Saver()

    for epoch in xrange(num_epochs): # each CIFAR10 batch run num_epoch times
      loss_history = []
      acc_history = []
      for cifar_batch_no in range(1,6): # one single CIFAR10 batch
        print 'Processing CIFAR batch number', cifar_batch_no
        f = os.path.join('cifar-10-batches-py', 'data_batch_'+str(cifar_batch_no))
        X, y = load_CIFAR_batch(f)
        X -= np.mean(X)
        X /= np.std(X)
        y = one_hot_vector(y, num_classes)
        if cifar_batch_no==1:
          X_validate = X[:batch_size]
          y_validate = y[:batch_size]
          X = X[batch_size:]
          y = y[batch_size:]
        print X.shape, y.shape
        num_steps = X.shape[0]/batch_size
        for step in xrange(num_steps): # each CIFAR10 batch divided into mini-batches
          X_batch, y_batch = X[step*batch_size:(step+1)*batch_size], y[step*batch_size:(step+1)*batch_size]
          # X_batch, y_batch = X[:batch_size], y[:batch_size] # sanity check; overfitting on small sample
          sess.run(optimizer, feed_dict={X_:X_batch, y_:y_batch, keep_prob:0.5})
          l, acc = sess.run([loss, accuracy], feed_dict={X_:X_batch, y_:y_batch, keep_prob:1.0})
          loss_history.append(l)
          acc_history.append(acc)
          print 'Epoch {}/{}, CIFAR batch number {}/{}, Iteration {}/{}, Loss {}, accuracy {}'.format(epoch, num_epochs, cifar_batch_no, 6, step, num_steps, np.mean(loss_history), np.mean(acc_history))
          if step%10==0:
            plt.plot(loss_history)
            plt.savefig('loss_graph.png')
            plt.close()
            plt.plot(acc_history)
            plt.savefig('acc_graph.png')
            plt.close()
          if step%100==0 and step>0:
            print 'Saving model'
            saver.save(sess, 'cifar_network.ckpt')

        # ========================== VALIDATION =====================================================
        l_validate, acc_validate = sess.run([loss, accuracy], feed_dict={X_:X_validate, y_:y_validate, keep_prob:1.0})
        print 'Validation loss = {}, Validation accuracy = {}'.format(l_validate, acc_validate)
      saver.save(sess, 'cifar_network.ckpt')

    # =================================== TESTING =========================================================
    X_test, y_test = load_CIFAR_batch(os.path.join('cifar-10-batches-py', 'test_batch'))
    y_test = one_hot_vector(y_test, num_classes)
    print X_test.shape, y_test.shape
    with tf.Session() as sess:
      sess.run(init)
      saver = tf.train.Saver()
      saver.restore(sess, '/home/arya_04/Tejash/DL_Practice/cifar_network.ckpt')
      num_steps = X_test.shape[0]/batch_size
      loss_history, acc_history = [], []
      for step in xrange(num_steps):
        X_test_batch, y_test_batch = X_test[step*batch_size:(step+1)*batch_size], y_test[step*batch_size:(step+1)*batch_size]
        predictions = cifar_network.build(X_test_batch, keep_prob)
        # print sess.run(predictions, feed_dict={X_:X_test_batch, y_:y_test_batch, keep_prob:.5})
        loss = cifar_network.calc_loss(predictions, y_test_batch)
        accuracy = cifar_network.calc_accuracy(predictions, y_test_batch)
        l, acc = sess.run([loss, accuracy], feed_dict={X_:X_test_batch, y_:y_test_batch, keep_prob:1})
        loss_history.append(l)
        acc_history.append(acc)
        print 'Testing Iter {}/{}, Loss {}, Acc {}'.format(step, num_steps, np.mean(loss_history), np.mean(acc_history))
