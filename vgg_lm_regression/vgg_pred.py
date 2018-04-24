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
trained_model_path="./tmp/ckpt/vgg/vgg_16_out_"+category_name+"_{}".format(imsize)+".ckpt"

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
    y_input = y_input[:,id_coords]

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
    lm_hat = tf.nn.sigmoid(logits_lm)
    vis_hat = tf.nn.sigmoid(logits_vis)
    loss_op,l_is_lm,l1,l2 = model.loss_all(coords_hat,logits_lm,logits_vis, Y )
    accuracy,accuracy_bool = model.acc_all(coords_hat,logits_lm,logits_vis, Y )

#optimizer 
ph_learning_rate = tf.placeholder(tf.float32, shape=[])
optimizer = tf.train.AdamOptimizer(learning_rate=ph_learning_rate)
train_op = optimizer.minimize(loss_op)

################# Session ##################
init = tf.global_variables_initializer()




restorer = tf.train.Saver()


with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    if load_var:
        print("Load model from path : " , trained_model_path)
        load_path = restorer.restore(sess, trained_model_path) 
    # results=sess.run(net, feed_dict={X:X_train})         

    results=sess.run(net, feed_dict={X:x_input})
    acc,acc_bool = sess.run([accuracy,accuracy_bool], feed_dict={conv_layer: results,
                                  Y: y_input})
    print("Testing Accuracy: coords:{:.3f}  bool:{:.3f}".format(acc,acc_bool)
    )

    pred_coords,pred_lm,pred_vis = sess.run([coords_hat,lm_hat,vis_hat], feed_dict={conv_layer: results,
                                      Y: y_coord})

    np.savetxt("./vgg_result.csv", a,fmt='%i', delimiter=",")

pred_coords=pred_coords*int(512/imsize)
pred_coords=pred_coords.astype(int)

pred_vis[pred_lm == 0] = -1

for i in range(lm_cnt):
    pred_coords = np.hstack((pred_coords[:,:(2)*(i+1)+i],pred_vis,pred_coords[:,(2)*(i+1)+i:]) )

import pandas as pd 
adf = pd.DataFrame(pred_coords)
out_data = pd.concat([pred_df,adf],axis=1)
out_data.to_csv("vgg_pred_"+category_name+"_"+im_size+".csv",index=False)