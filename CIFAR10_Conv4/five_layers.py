import cPickle as pickle
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt

# =============================== UTILITY FUNCTIONS ==========================================
def load_CIFAR_batch(filename):
  """ load single batch of cifar """
  with open(filename, 'rb') as f:
    datadict = pickle.load(f)
    X = datadict['data']
    Y = datadict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype(np.float32)
    Y = np.array(Y).astype(np.int32)
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

# =============================== CONVOLUTIONAL NETWORK =================================================
class CIFAR10Network:
  def __init__(self, num_classes):
    self.num_classes = num_classes 
    self.weights = {
      'wc1':tf.Variable(tf.random_normal([3, 3, 3, 32])),
      'wc2':tf.Variable(tf.random_normal([3, 3, 32, 64])),
      'wc3':tf.Variable(tf.random_normal([3, 3, 64, 128])),
      'wd1':tf.Variable(tf.random_normal([4*4*128, 160])),
      'wout':tf.Variable(tf.random_normal([160, num_classes]))
    }
    self.biases = {
      'bc1':tf.Variable(tf.random_normal([32])),
      'bc2':tf.Variable(tf.random_normal([64])),
      'bc3':tf.Variable(tf.random_normal([128])),
      'bd1':tf.Variable(tf.random_normal([160])),
      'bout':tf.Variable(tf.random_normal([num_classes]))
    }

  def conv2d(self, img, W, b, stride=1, padding='SAME', depth_radius=3):
    network = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, W, strides=[1,stride,stride,1], padding=padding), b))
    return tf.nn.local_response_normalization(network, depth_radius)

  def maxpool2d(self, img, k=1, padding='SAME'):
    return tf.nn.max_pool(img, ksize=[1,k,k,1], strides=[1,k,k,1], padding=padding)

  def calc_loss(self, logits, labels):
    # labels = np.argmax(labels, axis=1)
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))

  def calc_accuracy(self, predictions, y):
    correct_pred = tf.equal(tf.argmax(predictions, 1), y)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  def get_optimizer(self, loss, learning_rate=0.001):
    return tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

  def build(self, X, dropout=1, batch_size=32):
   
    conv1 = self.conv2d(X, self.weights['wc1'], self.biases['bc1'])
    conv1 = self.maxpool2d(conv1, k=2)
    conv1 = tf.nn.dropout(conv1, dropout)
    # print 'CONV1', conv1.get_shape()
    conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
    conv2 = self.maxpool2d(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, dropout)
    # print 'CONV2', conv2.get_shape()
    conv3 = self.conv2d(conv2, self.weights['wc3'], self.biases['bc3'])
    conv3 = self.maxpool2d(conv3, k=2)
    conv3 = tf.nn.dropout(conv3, dropout)
    # print 'CONV3', conv3.get_shape()
    fc1 = tf.reshape(conv3, [-1, conv3.get_shape().as_list()[0]])
    # print 'CONV4 reshaped', fc1.get_shape()
    fc1 = tf.add(tf.matmul(tf.transpose(fc1), self.weights['wd1']), self.biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)
    # print 'FC1', fc1.get_shape()
    out = tf.nn.relu(tf.add(tf.matmul(fc1, self.weights['wout']), self.biases['bout']))
    # print 'OUT', out.get_shape()
    return out

if __name__ == '__main__':
  num_classes = 10
  num_epochs = 50
  batch_size = 32
  height = 32
  width = 32
  num_channels = 3
  X_ = tf.placeholder(tf.float32, [batch_size, height, width, num_channels])
  y_ = tf.placeholder(tf.int32, [batch_size,])
  keep_prob = tf.placeholder(tf.float32)          
  correct_pred = tf.placeholder(tf.float32, [batch_size])
  cifar_network = CIFAR10Network(num_classes)
  init = tf.global_variables_initializer()

  # ================================ TRAINING ======================================================
  with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    try:
      saver.restore(sess, '/home/invenzone/tejash/DL_Practice/CIFAR10_Conv4/cifar_network.ckpt')
      print 'Restored model'
    except Exception, e:
      print 'Could not restore model'
    for epoch in xrange(num_epochs): # each CIFAR10 batch run num_epoch times
      loss_history = []
      acc_history = []
      for cifar_batch_no in range(1,6): # one single CIFAR10 batch
        print 'Processing CIFAR batch number', cifar_batch_no
        f = os.path.join('cifar-10-batches-py', 'test_batch')
        X, y = load_CIFAR_batch(f)
        X = X.astype(np.float32)
        # y = one_hot_vector(y, num_classes)
        y = y.astype(np.int32)
        print X.shape, y.shape, X.dtype, y.dtype
        num_steps = X.shape[0]/batch_size
        for step in xrange(num_steps): # each CIFAR10 batch divided into mini-batches
          # X_batch, y_batch = X[step*batch_size:(step+1)*batch_size], y[step*batch_size:(step+1)*batch_size]
          X_batch, y_batch = X[:batch_size], y[:batch_size] # sanity check; overfitting on small sample
          predictions = cifar_network.build(X_batch, keep_prob)
          # print sess.run(tf.argmax(predictions, 1), feed_dict={X_:X_batch, y_:y_batch, keep_prob:.5}), y_batch
          loss = cifar_network.calc_loss(predictions, y_batch)
          accuracy = cifar_network.calc_accuracy(predictions, y_batch)
          optimizer = cifar_network.get_optimizer(loss)
          l, acc, _ = sess.run([loss, accuracy, optimizer], feed_dict={X_:X_batch, y_:y_batch, keep_prob:.5})
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

      saver.save(sess, 'cifar_network.ckpt')

  # =================================== TESTING =========================================================
  # X_test, y_test = load_CIFAR_batch(os.path.join('cifar-10-batches-py', 'test_batch'))
  # y_test = one_hot_vector(y_test, num_classes)
  # print X_test.shape, y_test.shape
  # with tf.Session() as sess:
  #   sess.run(init)
  #   saver = tf.train.Saver()
  #   saver.restore(sess, '/home/invenzone/tejash/DL_Practice/CIFAR10_Conv4/cifar_network.ckpt')
  #   num_steps = X_test.shape[0]/batch_size
  #   loss_history, acc_history = [], []
  #   for step in xrange(num_steps):
  #     X_test_batch, y_test_batch = X_test[step*batch_size:(step+1)*batch_size], y_test[step*batch_size:(step+1)*batch_size]
  #     predictions = cifar_network.build(X_test_batch, keep_prob)
  #     # print sess.run(predictions, feed_dict={X_:X_test_batch, y_:y_test_batch, keep_prob:.5})
  #     loss = cifar_network.calc_loss(predictions, y_test_batch)
  #     accuracy = cifar_network.calc_accuracy(predictions, y_test_batch)
  #     l, acc = sess.run([loss, accuracy], feed_dict={X_:X_test_batch, y_:y_test_batch, keep_prob:1})
  #     loss_history.append(l)
  #     acc_history.append(acc)
  #     print 'Testing Iter {}/{}, Loss {}, Acc {}'.format(step, num_steps, np.mean(loss_history), np.mean(acc_history))
