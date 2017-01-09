import tensorflow as tf 
import numpy as np
from load_mnist import load_mnist

def next_batch(X, y, batch_size):
	n_items = y.shape[0]
	mask = np.random.randint(n_items, size=batch_size)
	return X[mask, :], y[mask]

def one_hot_vector(y_in, n_classes):
	n_items = y_in.shape[0]
	y_out = np.zeros((n_items, n_classes))
	y_out[np.arange(n_items), y_in[:, 0]] = 1
	return y_out

def conv2d(img, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(img, w, strides=[1,
		1,
		1,
		1], padding='SAME'), b))

def maxpool2d(img, k=1):
	return tf.nn.max_pool(img, ksize=[1,
		k,
		k,
		1], strides=[1,
		k,
		k,
		1], padding='SAME')

def lenet_network(X, y, dropout, mode='train', n_classes=10, learning_rate=0.001, batch_size=32):
	_variables = {
		'wc1':tf.Variable(tf.random_normal([5, 5, 1, 32])),
		'wc2':tf.Variable(tf.random_normal([5, 5, 32, 64])),
		'wd1':tf.Variable(tf.random_normal([7*7*64, 1024])),
		'wout':tf.Variable(tf.random_normal([1024, n_classes])),

		'bc1':tf.Variable(tf.random_normal([32])),
		'bc2':tf.Variable(tf.random_normal([64])),
		'bd1':tf.Variable(tf.random_normal([1024])),
		'bout':tf.Variable(tf.random_normal([n_classes]))
	}

	x = tf.placeholder(tf.float32, [None, 784])
	y_ = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
	conv1 = conv2d(X, _variables['wc1'], _variables['bc1'])
	print 'CONV1', conv1.get_shape()
	conv1 = maxpool2d(conv1, k=2)
	print 'CONV1 after maxpool', conv1.get_shape()
	conv2 = conv2d(conv1, _variables['wc2'], _variables['bc2'])
	print 'CONV2', conv2.get_shape()
	conv2 = maxpool2d(conv2, k=2)
	print 'CONV2 after maxpool', conv2.get_shape()

	fc1 = tf.reshape(conv2, [-1, _variables['wd1'].get_shape().as_list()[0]])
	fc1 = tf.add(tf.matmul(fc1, _variables['wd1']), _variables['bd1'])
	fc1 = tf.nn.relu(fc1)
	fc1 = tf.nn.dropout(fc1, dropout)
	print 'FC1', fc1.get_shape()

	# Output, class prediction
	out = tf.add(tf.matmul(fc1, _variables['wout']), _variables['bout'])

    # Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out, y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess, tf.device('/cpu:0'):
	    sess.run(init)
	    step = 0
	    # Keep training until reach max iterations
	    while step * batch_size < 2e6:
	        batch_x, batch_y = X[step:step+batch_size], y[step:step+batch_size]
	        batch_x = np.reshape(batch_x, (batch_x.shape[0], -1))
		        # Run optimization op (backprop)
	        sess.run(optimizer, feed_dict={x:batch_x, y_:batch_y, keep_prob:dropout})
	        if step % 10 == 0:
	            # Calculate batch loss and accuracy
	            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.})
	            print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
	                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
	                  "{:.5f}".format(acc)
	        step += 1
	    print "Optimization Finished!"

	    # Calculate accuracy for 256 mnist test images
	    print "Testing Accuracy:", \
	        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
	                                      y: mnist.test.labels[:256],
	                                      keep_prob: 1.})


if __name__=='__main__':
	images, labels = load_mnist(dataset='training')
	images = np.reshape(images, images.shape + (1,))
	print images.shape, labels.shape # should print (60000, 28, 28, 1) (60000, 1)
	labels = one_hot_vector(labels, 10)
	dropout = 0.75
	lenet_network(images, labels, dropout, mode='train', n_classes=10)