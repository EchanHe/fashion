""" Convolutional Neural Network.
Build and train a convolutional neural network with TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf

import train_input

import numpy as np

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Training Parameters
learning_rate = 0.001
num_steps = 50
batch_size = 1
display_step = 10
logs_path = './tmp/log/'
model_path = './tmp/ckpt/model1.ckpt'
# Network Parameters
imsize = 512
num_input = imsize*imsize*3 # MNIST data input (img shape: 28*28)
num_classes = 39 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
X = tf.placeholder(tf.float32, [None, num_input])
Y = tf.placeholder(tf.float32, [None, num_classes])
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
    # Reshape to match picture format [Height x Width x Channel]
    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
    x = tf.reshape(x, shape=[-1, imsize, imsize, 3])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1_1'], biases['bc1_1'])
    conv1 = conv2d(conv1, weights['wc1_2'], biases['bc1_2'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2_1'], biases['bc2_1'])
    conv2 = conv2d(conv2, weights['wc2_2'], biases['bc2_2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3_1'], biases['bc3_1'])
    conv3 = conv2d(conv3, weights['wc3_2'], biases['bc3_2'])
    conv3 = conv2d(conv3, weights['wc3_3'], biases['bc3_3'])
    # Max Pooling (down-sampling)
    conv3 = maxpool2d(conv3, k=2)


    conv4 = conv2d(conv3, weights['wc4_1'], biases['bc4_1'])
    conv4 = conv2d(conv4, weights['wc4_2'], biases['bc4_2'])
    conv4 = conv2d(conv4, weights['wc4_3'], biases['bc4_3'])
    # Max Pooling (down-sampling)
    conv4 = maxpool2d(conv4, k=2)

    conv5 = conv2d(conv4, weights['wc5_1'], biases['bc5_1'])
    conv5 = conv2d(conv5, weights['wc5_2'], biases['bc5_2'])
    conv5 = conv2d(conv5, weights['wc5_3'], biases['bc5_3'])
    # Max Pooling (down-sampling)
    conv5 = maxpool2d(conv5, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv5, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    fc2 = tf.nn.relu(fc2)
    # Apply Dropout
    fc2 = tf.nn.dropout(fc2, dropout)

    fc3 = tf.add(tf.matmul(fc2, weights['wd3']), biases['bd3'])
    fc3 = tf.nn.relu(fc3)
    # Apply Dropout
    fc3 = tf.nn.dropout(fc3, dropout)


    # Output, class prediction
    out = tf.add(tf.matmul(fc3, weights['out']), biases['out'])
    return out
##next batch function:
# def next_batch(X,y,batch_size ):
# def batch(X,y,batch_size):



# Store layers weight & bias
weights = {
    # 3x3 conv, 3 input, 64 outputs
    'wc1_1': tf.Variable(tf.random_normal([3, 3, 3, 64])),
    'wc1_2': tf.Variable(tf.random_normal([3, 3, 64, 64])),

    # 3x3 conv, 64 inputs, 128 outputs
    'wc2_1': tf.Variable(tf.random_normal([3, 3, 64, 128])),
    'wc2_2': tf.Variable(tf.random_normal([3, 3, 128, 128])),
    # 3x3 conv, 128 inputs, 256 outputs
    'wc3_1': tf.Variable(tf.random_normal([3, 3, 128, 256])),
    'wc3_2': tf.Variable(tf.random_normal([3, 3, 256, 256])),
    'wc3_3': tf.Variable(tf.random_normal([3, 3, 256, 256])),

    # 3x3 conv, 256 inputs, 512 outputs
    'wc4_1': tf.Variable(tf.random_normal([3, 3, 256, 512])),
    'wc4_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wc4_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),

    # 3x3 conv, 256 inputs, 512 outputs
    'wc5_1': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wc5_2': tf.Variable(tf.random_normal([3, 3, 512, 512])),
    'wc5_3': tf.Variable(tf.random_normal([3, 3, 512, 512])),


    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([int(imsize/32)*int(imsize/32)*512, 4096])),
    'wd2': tf.Variable(tf.random_normal([4096, 4096])),
    'wd3': tf.Variable(tf.random_normal([4096, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, num_classes]))
}

biases = {
    'bc1_1': tf.Variable(tf.random_normal([64])),
    'bc1_2': tf.Variable(tf.random_normal([64])),
    'bc2_1': tf.Variable(tf.random_normal([128])),
    'bc2_2': tf.Variable(tf.random_normal([128])),
    'bc3_1': tf.Variable(tf.random_normal([256])),
    'bc3_2': tf.Variable(tf.random_normal([256])),
    'bc3_3': tf.Variable(tf.random_normal([256])),
    'bc4_1': tf.Variable(tf.random_normal([512])),
    'bc4_2': tf.Variable(tf.random_normal([512])),
    'bc4_3': tf.Variable(tf.random_normal([512])),

    'bc5_1': tf.Variable(tf.random_normal([512])),
    'bc5_2': tf.Variable(tf.random_normal([512])),
    'bc5_3': tf.Variable(tf.random_normal([512])),

    'bd1': tf.Variable(tf.random_normal([4096])),
    'bd2': tf.Variable(tf.random_normal([4096])),
    'bd3': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Construct model
logits = conv_net(X, weights, biases, keep_prob)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

loss_op = tf.reduce_mean(tf.square(logits - Y)) / 2.

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.square(logits - Y)) / 2.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()





# x_input,y_cate = input_small.get_input(imsize,800)
x_input,y_cate = train_input.get_x_y(100,2)
import sklearn.model_selection as sk

X_train, X_test, y_train, y_test = sk.train_test_split(x_input,y_cate,test_size=0.1, random_state = 42)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

# Start training

print("X train shape: ", X_train.shape)
print("y train shape: ", y_train.shape)
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # load_path = saver.restore(sess, model_path)
    # print("Model restored from file: %s" % save_path)

    for step in range(1, num_steps+1):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        
        # print(type(batch_x))
        # print(batch_x.shape)
        # X_train, y_train = shuffle(X_train, y_train)
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        # print(type(batch_x))
        # print(batch_x.shape)
        for start_idx in range(0, X_train.shape[0] - batch_size + 1, batch_size):
        
            excerpt = indices[start_idx:start_idx + batch_size]
            # Get pair of (X, y) of the current minibatch/chunk
            X_train_mini = X_train[excerpt]
            y_train_mini = y_train[excerpt]



            sess.run(train_op, feed_dict={X: X_train_mini, Y: y_train_mini, keep_prob: 0.8})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: X_train_mini,
                                                                     Y: y_train_mini,
                                                                     keep_prob: 1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    # save_path = saver.save(sess, model_path)                
    print("Optimization Finished!")

    # Calculate accuracy for 256 MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: X_test,
                                      Y: y_test,
                                      keep_prob: 1.0}))