"""
vgg train on coords

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf

slim = tf.contrib.slim
import train_input
import vgg

import numpy as np
import sklearn.model_selection as sk


imsize = 128
learning_rate = 0.001
##steps and batch
batch_size = 100
num_steps =1000
display_step = 5

save_log=False
###Deal with input image
x_input,y_input = train_input.get_x_y(110,512/imsize, flat_x = False)

print("input X shape" ,  x_input.shape)


data_cols = y_input.shape[1]
lm_cnt =int( y_input.shape[1]/4)
id_coords = np.arange(0, lm_cnt*2)
id_islm = np.arange(lm_cnt*2, lm_cnt*3)
id_vis = np.arange(lm_cnt*3, lm_cnt*4)
y_lm = y_input[:,id_islm]
y_vis = y_input[:,id_vis]

y_coord = y_input[:,id_coords]

X_train, X_test, y_train, y_test = sk.train_test_split(x_input,y_coord,test_size=0.1, random_state = 42)


logs_path="./tmp/"
model_path="./vgg_16.ckpt"

X = tf.placeholder(tf.float32, [None, imsize,imsize,3])


net = vgg.vgg_16_with_img_size (X,
      num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID',
           global_pool=False,
           im_size=128
           )


print("vgg net output shape: " , net.shape)


weights = {

    # 1024 inputs, 10 outputs (class prediction)
    'out1': tf.Variable(tf.random_normal([4096, lm_cnt*2]))
    }


biases= {

    'out1': tf.Variable(tf.random_normal([lm_cnt*2]))
    # 'out2': tf.Variable(tf.random_normal([num_classes]))
}



vgg_fc_out = tf.placeholder(tf.float32, [None,4096])
Y = tf.placeholder(tf.float32, [None, lm_cnt*2])

coords_hat = tf.add(tf.matmul(vgg_fc_out, weights['out1']), biases['out1'])

loss_op = tf.reduce_mean(tf.square(coords_hat - Y)) / 2.

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

accuracy = tf.reduce_mean(tf.sqrt(tf.square(coords_hat - Y))) / 2.

init = tf.global_variables_initializer()

#only save vars named with "vgg_16"
variables_to_restore = slim.get_variables(scope="vgg_16")
for var in variables_to_restore:
    print(var)
saver = tf.train.Saver(variables_to_restore)
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    if save_log:
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # load_path = saver.restore(sess, model_path) 
    # results=sess.run(net, feed_dict={X:X_train})
    
    # for var in tf.trainable_variables():
    #     print (var.name , var.shape)



    for step in range(1, num_steps+1):

        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        # print(type(batch_x))
        # print(batch_x.shape)
        for start_idx in range(0, X_train.shape[0] - batch_size + 1, batch_size):
        
            excerpt = indices[start_idx:start_idx + batch_size]
            # Get pair of (X, y) of the current minibatch/chunk
            X_train_mini = X_train[excerpt]
            y_train_mini = y_train[excerpt]
            results_mini=sess.run(net, feed_dict={X:X_train_mini})
            sess.run(train_op, feed_dict={vgg_fc_out: results_mini, Y: y_train_mini})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={vgg_fc_out: results_mini,
                                                                     Y: y_train_mini})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
    # save_path = saver.save(sess, model_path)   

    results=sess.run(net, feed_dict={X:X_test})
    print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={vgg_fc_out: results,
                                  Y: y_test}))
    a,acc = sess.run([coords_hat,accuracy], feed_dict={vgg_fc_out: results,
                                      Y: y_test})

    np.savetxt("./vgg_result.csv", a,fmt='%i', delimiter=",")


