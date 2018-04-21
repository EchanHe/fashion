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
import categories
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
            x = map_shape[1]
            y = map_shape[0]
            output_result[i,j*2+0] = x*scale
            output_result[i,j*2+1] = y*scale
            if coord[i,j*2] ==-1:
                output_result[i,j*2+0] = -1
                output_result[i,j*2+1] = -1
#             print("x pred: {} origin: {}".format(x*scale,coord[i,j*2]))
#             print("y pred: {} origin: {}".format(y*scale,coord[i,j*2+1]))
    #print(output_result)
    acc_per_row = np.sqrt(np.mean(np.square(output_result-coord) , axis =1))
    #print("\tAccuracy: "+str(acc_per_row.astype(int)))
    return acc_per_row
    
    

class Config():
    
    def __init__(self,category,imsize , learning_rate, lambda_l2=0.0 , keep_prob = 1.0):
        self.pre_path = "./"
        self.category = category
        self.img_height = imsize
        self.img_width = imsize
        self.load_filename = "cpm_"+category+".ckpt"
        self.save_filename = "cpm_"+category+".ckpt"
        
        self.fm_width = self.img_width >> 1
        self.fm_height = self.img_height >> 1
        
        self.learning_rate = learning_rate

        #Regularzation:
        self.lambda_l2 = lambda_l2
        self.keep_prob = keep_prob        
                
        folder = "l2_{}_drop_{}/".format(self.lambda_l2 , self.keep_prob) 
        self.logdir =  os.path.join(self.pre_path+"/log/train_log/CPM/", category , folder)
        self.params_dir = os.path.join(self.pre_path+"/params/CPM/" , category , folder)
    # =================== modify parameters ==================
    TAG = "_demo" # used for uniform filename
               # "_demo": train with demo images
               # "": (empty) train with ~60000 images
    batch_size = 5
    initialize = True # True: train from scratch (should also
                    # delete the corresponding params files
                    # in params_dir);
                     # False: restore from pretrained model
    steps = "30000"   # if 'initialize = False', set steps to 
                     # where you want to restore
    toDistort = False
    # iterations config
    max_iteration = 40000
    checkpoint_iters = 5000
    summary_iters = 100
    validate_iters = 4000
    
    
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

    lambda_l2 = 0.001
    # solver configinference_person
    wd = 5e-4
    stddev = 5e-2
    use_fp16 = False
    moving_average_decay = 0.999


l = 1e-04
lambda_l2 = 0.001
keep_prob= 0.5
args = sys.argv
print(args)
category_name = args[1]
imsize = int(args[2])
total_size =int( args[3])


#imsize = 512
#total_size = -1
test_size =200
#category_name = "trousers"

config = Config(category_name,imsize,l,lambda_l2,keep_prob)
config.points_num = categories.get_cate_lm_cnts(category_name)


print("steps: {}\nlearning rates: {}".format(config.max_iteration ,config.learning_rate))

file_name = "./train_pad/Annotations/train_"+category_name+"_coord_with_warm_wrong.csv"
ckpt_file_name = "/data/bop16yh/fashion/params/CPM/" + category_name +"/cpm_"+category_name + ".ckpt-20000"


is_retrain = False
print("read data from: "+file_name)
print("====RETRAIN ?: {}==== \n  Checkpoint file from: {}".format(is_retrain,ckpt_file_name))
df = pd.read_csv(file_name)
if total_size>-1:
    df = df[:]
input_data = train_input.data_stream(df,config.batch_size,is_train=True)


df_test = pd.read_csv("train_pad/Annotations/train_"+category_name+"_coord_valid.csv")
# df_test = df_test[:config.batch_size]
test_data = train_input.data_stream(df_test,config.batch_size,is_train=True)
# x_input,y_input,coords_input,maps_input = train_input.get_x_y_map(total_size,512/imsize, cates = category_name, flat_x = False)

# input_data = train_input.data2(x_input,y_input,coords_input,maps_input,config.batch_size,is_train=True)

# x_test, y_test,coord_test,center_test  = train_input.get_x_y_map_valid(test_size,512/imsize,pre_dir="./train_warm_up_pad/", cates = category_name, flat_x = False)
# test_data = train_input.data2(x_test,y_test,coord_test,center_test,config.batch_size,is_train=True)

###model####


tf.reset_default_graph()
#with tf.device('/device:GPU:0'):
print("==========lambda: {}   drop: {}========".format(config.lambda_l2 , config.keep_prob))
model = cpm.CPM(config)
predict = model.inference_pose_vgg_l2()

loss = model.loss()
train_op = model.train_op(loss, model.global_step)
# accuracy_model = model.accuracy()



# file_name = "./train_pad/Annotations/train_"+category_name+"_coord_with_warm.csv"
# ckpt_file_name = "/data/bop16yh/fashion/params/CPM/" + category_name +"/cpm_"+category_name + ".ckpt-20000"


saver = tf.train.Saver()




init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    # sess.run(init_op)

    if not os.path.exists(config.params_dir):
        os.makedirs(config.params_dir)
    if os.listdir(config.params_dir) == [] or config.initialize:
        print ("Initializing Network")
        sess.run(init_op)
    else:
        sess.run(init_op)
        model.restore(sess, saver, config.load_filename)
    if is_retrain:
        saver.restore(sess , ckpt_file_name)
    merged = tf.summary.merge_all()
    logdir = config.logdir

    writer = tf.summary.FileWriter(logdir, sess.graph)


    valid_loss_ph = tf.placeholder(tf.float32, shape=(), name="valid_loss")
    loss_valid = tf.summary.scalar('loss_valid',valid_loss_ph ) 
    valid_acc_ph = tf.placeholder(tf.float32, shape=(), name="valid_acc")
    acc_valid = tf.summary.scalar('acc_valid',valid_acc_ph ) 

    for i in range(config.max_iteration):

        X_train_mini,y_train_mini,coords_mini,vis_mini,center_mini,_ = input_data.get_next_batch_no_random()
        feed_dict = {
                    model.images: X_train_mini,
                    model.labels: y_train_mini,
                    model.vis_mask: vis_mini,
                    model.keep_prob:config.keep_prob
                    }
        sess.run(train_op, feed_dict=feed_dict)
        if (i+1) % config.summary_iters == 0:
#             print("accuracy: {}".format(sess.run(accuracy_model, feed_dict=feed_dict)))
#             print("accuracy: {}".format(len(sess.run(accuracy_model, feed_dict=feed_dict))))
            print("{} steps Loss: {}".format(i,sess.run(loss, feed_dict=feed_dict)))

            result=sess.run(predict, feed_dict=feed_dict)
            acc = accuracy(result,coords_mini,imsize)
            print("\tAccuracy: {}".format(acc.astype(int)))

            tmp_global_step = model.global_step.eval()
            lear = model.learning_rate.eval()
            print("\tGlobal steps and learning rates: {}  {}".format(tmp_global_step,lear))


            summary = sess.run(merged, feed_dict=feed_dict)    
            writer.add_summary(summary, tmp_global_step)
            writer.flush()

        if (i+1) % config.validate_iters== 0:
            acc_list = np.array([])
            loss_list = np.array([])
            for i_df_test in np.arange(0,df_test.shape[0],config.batch_size):
                X_mini,y_mini,coords_mini,vis_mini,center_mini , _ = test_data.get_next_batch_no_random()
                feed_dict = {
                        model.images: X_mini,
                        model.labels: y_mini,
                        model.vis_mask: vis_mini,
                        model.keep_prob: 1.0
                        }
                #print("\t VALIDATION: {} steps Loss: {}".format(i,sess.run(loss, feed_dict=feed_dict)))
                
                _loss = sess.run(loss, feed_dict=feed_dict)

                result_mini = sess.run(predict, feed_dict=feed_dict)
                acc = accuracy(result_mini,coords_mini,imsize)
                
                acc_list = np.concatenate((acc_list,acc))
                loss_list = np.append(loss_list,_loss)

            print("\t VALIDATION {} steps: average acc and loss : {}  {}".format(i,np.mean(acc_list),np.mean(loss_list)))
            writer.add_summary(sess.run(loss_valid, feed_dict={valid_loss_ph: np.mean(loss_list)}) , i)  
            writer.add_summary(sess.run(acc_valid, feed_dict={valid_acc_ph: np.mean(acc_list)}) , i) 
        if (i + 1) % config.checkpoint_iters == 0:
            tmp_global_step = model.global_step.eval()
            model.save(sess, saver, config.save_filename,tmp_global_step)
    
#file_name = "/data/bop16yh/fashion/params/CPM/trousers/cpm_trousers.ckpt-12200"
    
    


