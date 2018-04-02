from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf

slim = tf.contrib.slim

# import vgg
# import vgg16_coord_model as model
import numpy as np
import pandas as pd
import sklearn.model_selection as sk
from enum import Enum
import sys
import os
import time

import importlib
import train_input
import cpm
importlib.reload(train_input)
importlib.reload(cpm)



def accuracy(result, coord, imsize):
    df_size = result.shape[0]
    cnt_size = result.shape[3]
    
    scale = (512/(imsize/8))
    output_result = np.ones((df_size,cnt_size*2))
    for i in range(df_size):
        for j in range(cnt_size):
            heat_map = result[i,:,:,j]
            map_shape = np.unravel_index(np.argmax(heat_map, axis=None), heat_map.shape)
            x = map_shape[1]+1
            y = map_shape[0]+1
            output_result[i,j*2+0] = x*scale
            output_result[i,j*2+1] = y*scale
#             print("x pred: {} origin: {}".format(x*scale,coord[i,j*2]))
#             print("y pred: {} origin: {}".format(y*scale,coord[i,j*2+1]))
    #print(output_result)
    acc_per_row = np.sqrt(np.mean(np.square(output_result-coord) , axis =1))
    print( acc_per_row)
    
    

class Config():
    
    def __init__(self,category,imsize , learning_rate):
        self.category = category
        self.img_height = imsize
        self.img_width = imsize
        self.load_filename = "cpm_"+category+".ckpt"
        self.save_filename = "cpm_"+category+".ckpt"
        
        self.fm_width = self.img_width >> 1
        self.fm_height = self.img_height >> 1
        
        self.learning_rate = learning_rate
                #file path
        self.logdir =  os.path.join("./log/train_log/CPM/", category , str(self.learning_rate))
        self.params_dir = "./params/CPM/" + category + "/"
    # =================== modify parameters ==================
    TAG = "_demo" # used for uniform filename
               # "_demo": train with demo images
               # "": (empty) train with ~60000 images
    batch_size = 2
    initialize = True # True: train from scratch (should also
                    # delete the corresponding params files
                    # in params_dir);
                     # False: restore from pretrained model
    steps = "30000"   # if 'initialize = False', set steps to 
                     # where you want to restore
    toDistort = False
    # iterations config
    max_iteration = 1000
    checkpoint_iters = 2000
    summary_iters = 100
    validate_iters = 2000
    
    
#     category
    # ========================================================

#     annos_path = "./labels/txt/input/train_annos" + TAG + ".txt"
#     data_path = "./data/input/train_imgs" + TAG + "/"
    gpu = '/gpu:0'

    # checkpoint path and filename
#     logdir = "./log/train_log/"
#     params_dir = "./params/" + TAG + "/"
    load_filename = "cpm" + '-' + steps

    
    # image config
    points_num = 7
    fm_channel = points_num + 1
    #   origin_height = 212
    #   origin_width = 256
    

    # feature map config

    sigma = 2.0
    alpha = 1.0
    radius = 12

    # random distortion
    degree = 8

    # solver configinference_person
    wd = 5e-4
    stddev = 5e-2
    use_fp16 = False
    moving_average_decay = 0.999


l = 5e-06


imsize = 512
total_size = -1
test_size =30
category_name = "trousers"

config = Config(category_name,imsize,l)

print("read data from: "+"train_pad/Annotations/train_"+category_name+"_coord.csv")
print("steps: {}\nlearning rates: {}".format(config.max_iteration ,config.learning_rate))

df = pd.read_csv("train_pad/Annotations/train_"+category_name+"_coord.csv")
df = df[:10]
input_data = train_input.data_stream(df,config.batch_size,is_train=True)



# x_input,y_input,coords_input,maps_input = train_input.get_x_y_map(total_size,512/imsize, cates = category_name, flat_x = False)

# input_data = train_input.data2(x_input,y_input,coords_input,maps_input,config.batch_size,is_train=True)

# x_test, y_test,coord_test,center_test  = train_input.get_x_y_map_valid(test_size,512/imsize,pre_dir="./train_warm_up_pad/", cates = category_name, flat_x = False)
# test_data = train_input.data2(x_test,y_test,coord_test,center_test,config.batch_size,is_train=True)

###model####



print("learning rate: {}".format(l))
tf.reset_default_graph()

model = cpm.CPM(config)
predict = model.inference_pose_vgg()

loss = model.loss()
train_op = model.train_op(loss, model.global_step)
# accuracy_model = model.accuracy()


all_var = tf.trainable_variables()

saver = tf.train.Saver()




init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)

    if not os.path.exists(config.params_dir):
        os.makedirs(config.params_dir)
    if os.listdir(config.params_dir) == [] or config.initialize:
        print ("Initializing Network")
        sess.run(init_op)
    else:
        sess.run(init_op)
        model.restore(sess, saver, config.load_filename)
    merged = tf.summary.merge_all()
    logdir = config.logdir

    writer = tf.summary.FileWriter(logdir, sess.graph)
    for i in range(config.max_iteration):

        X_train_mini,y_train_mini,coords_mini,center_mini = input_data.get_next_batch()
        feed_dict = {
                    model.images: X_train_mini,
                    model.labels: y_train_mini,
                    }
        sess.run(train_op, feed_dict=feed_dict)
        if (i + 1) % config.summary_iters == 0:
#             print("accuracy: {}".format(sess.run(accuracy_model, feed_dict=feed_dict)))
#             print("accuracy: {}".format(len(sess.run(accuracy_model, feed_dict=feed_dict))))
            print("{} steps Loss: {}".format(i,sess.run(loss, feed_dict=feed_dict)))
            result=sess.run(predict, feed_dict=feed_dict)
            accuracy(result,coords_mini,imsize)
        
            tmp_global_step = model.global_step.eval()
            summary = sess.run(merged, feed_dict=feed_dict)    
            writer.add_summary(summary, tmp_global_step)
            writer.flush()

    #abs
#     feed_dict = {
#         model.images: x_test,
#         model.labels: y_test
#     }
#     print("Test loss: {}".format(sess.run(predict, feed_dict=feed_dict) ))
    tmp_global_step = model.global_step.eval()
    model.save(sess, saver, config.save_filename,tmp_global_step)