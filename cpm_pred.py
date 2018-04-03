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
from scipy.ndimage import gaussian_filter
import sys


class Config():
    
    def __init__(self,category,imsize , learning_rate):
        self.pre_path = "/data/bop16yh/fashion"
        self.category = category
        self.img_height = imsize
        self.img_width = imsize
        self.load_filename = "cpm_"+category+".ckpt"
        self.save_filename = "cpm_"+category+".ckpt"
        
        self.fm_width = self.img_width >> 1
        self.fm_height = self.img_height >> 1
        
        self.learning_rate = learning_rate
                #file path
        self.logdir =  os.path.join(self.pre_path+"/log/train_log/CPM/", category , str(self.learning_rate))
        self.params_dir = self.pre_path+"/params/CPM/" + category + "/"
    # =================== modify parameters ==================
    TAG = "_demo" # used for uniform filename
               # "_demo": train with demo images
               # "": (empty) train with ~60000 images
    batch_size = 20
    initialize = True # True: train from scratch (should also
                    # delete the corresponding params files
                    # in params_dir);
                     # False: restore from pretrained model
    steps = "30000"   # if 'initialize = False', set steps to 
                     # where you want to restore
    toDistort = False
    # iterations config
    max_iteration = 5000
    checkpoint_iters = 500
    summary_iters = 100
    validate_iters = 300
    
    
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
    points_num = 4
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

args = sys.argv



print(args)
category_name = args[1]
imsize = int(args[2])
total_size =int( args[3])

scale = int(512/imsize)
config = Config(category_name,imsize,0.1)

pre_path = "./test_pad/"
file_name = "~/fashion/test_pad/test_"+category_name+".csv"
ckpt_file_name = "/data/bop16yh/fashion/params/CPM/trousers/cpm_trousers.ckpt"

out_data_folder = "/data/bop16yh/fashion/result/"
if not os.path.exists(out_data_folder):
    os.makedirs(out_data_folder)
out_data_path = out_data_folder+"cpm_"+category_name+"_"+str(imsize)+".csv"

print("read data from: "+file_name)

df = pd.read_csv(file_name)
if total_size>-1:
    df = df[:total_size]
pred_df = df[["image_id","image_category"]]
input_data = train_input.data_stream(df,config.batch_size,is_train=False,scale = scale ,pre_path ="./test_pad/")


tf.reset_default_graph()
#with tf.device('/device:GPU:0'):
model = cpm.CPM(config)
predict = model.inference_pose_vgg(is_train=False)

saver = tf.train.Saver(tf.trainable_variables())

init_op = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init_op)

#     if not os.path.exists(config.params_dir):
#         os.makedirs(config.params_dir)
#     if os.listdir(config.params_dir) == [] or config.initialize:
    print ("Initializing Network")
#         sess.run(init_op)from scipy.ndimage import gaussian_filter
#     else:
#         sess.run(init_op)
#         model.restore(sess, saver, config.load_filename)
#
    saver.restore(sess , ckpt_file_name)
#     merged = tf.summary.merge_all()
#     logdir = config.logdir

#     writer = tf.summary.FileWriter(logdir, sess.graph)

    for i in np.arange(0,df.shape[0],config.batch_size):
    
        X_mini = input_data.get_next_batch_no_random_all()
        feed_dict = {
                    model.pred_images: X_mini
                    }
        result_mini = sess.run(predict,feed_dict=feed_dict)
        if i ==0:
            result = result_mini
        else:
            result = np.vstack((result,result_mini))
        print(result.shape)

df_size = result.shape[0]
cnt_size = result.shape[3]

output_result = np.ones((df_size,cnt_size*3))

scale = (512/(imsize/8-1))

for i in range(df_size):
    for j in range(cnt_size):
        heat_map = result[i,:,:,j]
        map_shape = np.unravel_index(np.argmax(heat_map, axis=None), heat_map.shape)
        x = map_shape[1]+1
        y = map_shape[0]+1
        output_result[i,j*3+0] = x*scale
        output_result[i,j*3+1] = y*scale
output_result = output_result.astype(int)
result_pd = pd.DataFrame(output_result)

out_data = pd.concat([pred_df,result_pd],axis=1)
out_data.to_csv(out_data_path,index=False)
print("write the result in: "+ out_data_path)