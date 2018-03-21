from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf

slim = tf.contrib.slim
import train_input
import vgg

import numpy as np
import vgg16_coord_model as model
import sklearn.model_selection as sk
from enum import Enum

import sys

import time
start_time = time.time()

class MODE(Enum):
    TRAIN = 1
    VALID = 2
    PRED = 3


imsize = 128
learning_rate = 0.005
##steps and batch
category_name = "skirt"

total_size = -1
test_percent=0.001
batch_size = 30
num_steps =40
display_step = 5
dropout = 0.75

save_log=True
load_var = True

retrain=False
flag_only_coord = False


######
args = sys.argv
print(args)
category_name = args[1]
imsize = int(args[2])
total_size =int( args[3])
#####
logs_path="./tmp/"
model_path="./vgg_16.ckpt"
output_model_path="./tmp/ckpt/vgg/vgg_16_out_"+category_name+"_{}".format(imsize)+".ckpt"

###Deal with input image
x_input,y_input = train_input.get_x_y(total_size,512/imsize, cates = category_name, flat_x = False)
print("--- %s mins reading data ---" % ((time.time() - start_time)/60.0))
start_time=time.time()
# x_input,y_input = train_input.get_x_y_s_e(6000,6200,scale=512/imsize,pre_dir="train_pad/",cates=category_name,flat_x = False)
print("##############")
print("Image Size: {} \n\nCategory: {}\ntotal_size: {}\nbatch_size: {}\nsteps: {}".format(imsize,category_name,total_size , batch_size,num_steps))
print("##############")
print("input X shape" ,  x_input.shape)


data_cols = y_input.shape[1]
lm_cnt =int( y_input.shape[1]/4)

id_coords = np.arange(0, lm_cnt*2)
id_islm = np.arange(lm_cnt*2, lm_cnt*3)
id_vis = np.arange(lm_cnt*3, lm_cnt*4)
y_lm = y_input[:,id_islm]
y_vis = y_input[:,id_vis]
if flag_only_coord:
    y_coord = y_input[:,id_coords]
    X_train, X_test, y_train, y_test = sk.train_test_split(x_input,y_coord,test_size=test_percent, random_state = 42)

    X = tf.placeholder(tf.float32, [None, imsize,imsize,3])
    net = vgg.vgg_16_with_img_size (X,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='vgg_16',
               fc_conv_padding='VALID',
               global_pool=False,
               im_size=imsize
               )

    conv_layer=tf.placeholder(tf.float32, [None,int(imsize/32)*int(imsize/32) *512])
    keep_prob = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32, [None, lm_cnt*2])
    fc_layer = model.fc_layers(conv_layer,keep_prob=keep_prob,imsize = imsize,lm_cnt =lm_cnt)
    loss_op = model.loss(fc_layer,Y)

    accuracy = tf.reduce_mean(tf.sqrt(tf.square(fc_layer - Y))) 
else:
    X_train, X_test, y_train, y_test = sk.train_test_split(x_input,y_input,test_size=test_percent, random_state = 42)

    X = tf.placeholder(tf.float32, [None, imsize,imsize,3])
    net = vgg.vgg_16_with_img_size (X,
               is_training=True,
               dropout_keep_prob=0.5,
               spatial_squeeze=True,
               scope='vgg_16',
               fc_conv_padding='VALID',
               global_pool=False,
               im_size=imsize
               )

    conv_layer=tf.placeholder(tf.float32, [None,int(imsize/32)*int(imsize/32) *512])
    keep_prob = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32, [None, lm_cnt*4])

    coords_hat, logits_lm , logits_vis = model.fc_layers_all_1_layer(conv_layer,keep_prob=keep_prob,imsize = imsize,lm_cnt =lm_cnt)
    loss_op,l_is_lm,l1,l2 = model.loss_all(coords_hat,logits_lm,logits_vis, Y )
    accuracy,accuracy_bool = model.acc_all(coords_hat,logits_lm,logits_vis, Y )

#optimizer 
ph_learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate=ph_learning_rate)
train_op = optimizer.minimize(loss_op)

################# Session ##################
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", loss_op)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", accuracy)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()


#only save vars named with "vgg_16"
if retrain==False:
    variables_to_restore = slim.get_variables(scope="vgg_16")
    restorer = tf.train.Saver(variables_to_restore)
else:
    restorer=tf.train.Saver()

saver = tf.train.Saver()
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    if save_log:
        summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    if load_var:
        if retrain==False:
            print("###################")
            print("Load model from path : " , model_path)
            variables_to_restore = slim.get_variables(scope="vgg_16")
            restorer.restore(sess, model_path) 
        else:
            restorer.restore(sess, output_model_path) 
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
            sess.run(train_op, feed_dict={conv_layer: results_mini, 
                                        Y: y_train_mini,
                                        ph_learning_rate:learning_rate,
                                        keep_prob:dropout})

            summary = sess.run(merged_summary_op, feed_dict={conv_layer: results_mini,
                                                                 Y: y_train_mini,
                                                                 keep_prob:1.0})
            summary_writer.add_summary(summary, step * X_train.shape[0]/batch_size + start_idx)
                                                                 
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            results_mini=sess.run(net, feed_dict={X:X_train_mini})
            if flag_only_coord:
                loss, acc,co = sess.run([loss_op, accuracy,coords_hat], feed_dict={conv_layer: results_mini,
                                                                     Y: y_train_mini,
                                                                     keep_prob:1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
            else:
                loss, acc,acc_bool,co = sess.run([loss_op, accuracy,accuracy_bool,coords_hat], feed_dict={conv_layer: results_mini,
                                                                     Y: y_train_mini,
                                                                     keep_prob:1.0})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc) +", bool acc= "+\
                    "{:.3f}".format(acc_bool))
            is_lm_v,l_is_lm_v,l_vis,l_lm = sess.run([Y[:,lm_cnt*2:lm_cnt*3] ,l_is_lm,l1,l2] ,
                        feed_dict = {conv_layer: results_mini,
                                                         Y: y_train_mini,
                                              keep_prob:1.0} )
            # print("Loss: is lm: {:.3f}, is vis: {:.3f} landmarks:{:.3f}".format(l_is_lm_v,l_vis,l_lm))
            print("Loss: is lm: {:.3f}, landmarks:{:.3f}".format(l_is_lm_v,l_lm))
            print("------")
            print(np.sum(y_train_mini>=0))
            print(np.sum(co>=0))
            print(np.sum(co<0))
            print("------")

                # print(is_lm_v)
                # print(l_vis)
                # print(l_lm)
        # if step==60:
        #     learning_rate=0.00005
        # if step==100:
        #     learning_rate=0.00001
        # if step==150:
        #     learning_rate=0.000001

    print("Model save in path : " , output_model_path)            
    save_path = saver.save(sess, output_model_path)   
    results=sess.run(net, feed_dict={X:X_test})
    print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={conv_layer: results,
                                  Y: y_test,
                                  keep_prob:1.0}), 
            sess.run(accuracy_bool, feed_dict={conv_layer: results,
                                      Y: y_test,
                                      keep_prob:1.0}))
    a,acc = sess.run([coords_hat,accuracy], feed_dict={conv_layer: results,
                                      Y: y_test,
                                      keep_prob:1.0})


    np.savetxt("./vgg_result"+str(imsize)+".csv", a,fmt='%i', delimiter=",")
print("--- %s mins runing model  ---" % ((time.time() - start_time)/60.0))