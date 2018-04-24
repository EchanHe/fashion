from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf

slim = tf.contrib.slim

# import vgg
# import vgg16_coord_model as model
import numpy as np
import pandas as pd

import sys
import os


import train_input
import cpm
from scipy.ndimage import gaussian_filter
import sys

dirname = os.path.dirname(__file__)
absdir =  os.path.abspath(dirname)
sys.path.append(os.path.join(dirname, '../util'))
import categories


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("cate", choices=["blouse" ,"outwear","trousers","skirt","dress" ],
                    help="The clothes category")
parser.add_argument("imsize", choices=["128","256","512" ],
                    help="Image size for training")
parser.add_argument("total_size", type = int,help="Training set size")
parser.add_argument("-l", "--learning_rate", type=float, default =1e-04,        
                    help="learning rate")
parser.add_argument("-l2","--lambda_l2", type=float, default = 0.001  ,
                    help="lambda for L2 regularization")
parser.add_argument("-k", "--keep_prob", type=float, default = 0.5  ,             
                    help="keep probability for drop out")

class Config():
    """
    训练模型的配置类
    """
    def __init__(self,category,imsize , learning_rate, lambda_l2=0.0 , keep_prob = 1.0):
        self.pre_path = os.path.join(dirname , "../")
        self.category = category
        self.img_height = imsize
        self.img_width = imsize
        
        self.save_filename = "cpm_"+category+".ckpt"
        
        self.fm_width = self.img_width >> 1
        self.fm_height = self.img_height >> 1
        
        self.learning_rate = learning_rate

        #Regularzation:
        self.lambda_l2 = lambda_l2
        self.keep_prob = keep_prob        
                
        folder = "l2_{}_drop_{}/".format(self.lambda_l2 , self.keep_prob) 
        self.logdir =  os.path.abspath(os.path.join(self.pre_path+"/log/train_log/CPM/", category , folder))
        self.params_dir = os.path.abspath(os.path.join(self.pre_path+"/params/CPM/" , category , folder))+"/"
        self.load_filename = "cpm_"+category + ".ckpt-50"
    # =================== modify parameters ==================
    TAG = "_demo" # used for uniform filename
               # "_demo": train with demo images
               # "": (empty) train with ~60000 images
    batch_size = 5
    initialize = False # True: train from scratch (should also
                    # delete the corresponding params files
                    # in params_dir);
                     # False: restore from pretrained model
    steps = "30000"   # if 'initialize = False', set steps to 
                     # where you want to restore
    # iterations config
    max_iteration = 5000
    checkpoint_iters = 500
    summary_iters = 100
    validate_iters = 300

    
    # image config
    points_num = 4

#命令行参数赋值 配置类
args = parser.parse_args()
print(args)

l = args.learning_rate
lambda_l2 = args.lambda_l2
keep_prob= args.keep_prob
category_name = args.cate
imsize = int(args.imsize)
total_size = int(args.total_size)

config = Config(category_name,imsize,l,lambda_l2,keep_prob)
config.points_num = categories.get_cate_lm_cnts(category_name)
# print("land mark point : {}".format(config.points_num))

pre_path = "./test_pad/"

file_name = os.path.abspath(os.path.join(absdir , "../test_pad/test_"+category_name+".csv"))



df = pd.read_csv(file_name)
if total_size>-1:
    df = df[:total_size]
pred_df = df[["image_id","image_category"]]
input_data = train_input.data_stream(df,config.batch_size,is_train=False,scale =  int(512/imsize) ,pre_path ="./test_pad/")

print("read data from: "+file_name)
##模型###
model = cpm.CPM(config)
predict = model.inference_pose_vgg(is_train=False)

saver = tf.train.Saver(tf.trainable_variables())

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)

    if not os.path.exists(config.params_dir) or os.listdir(config.params_dir) == [] or config.initialize:
        # print ("The checkpoint file is not")
        raise Exception("The checkpoint file is not exists")
        sess.run(init_op)
    else:
        sess.run(init_op)
        model.restore(sess, saver, config.load_filename)

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


out_data_folder = os.path.abspath(os.path.join(absdir , "../result"))
if not os.path.exists(out_data_folder):
    os.makedirs(out_data_folder)
out_data_path = out_data_folder+"/cpm_"+category_name+"_"+str(imsize)+".csv"


df_size = result.shape[0]
cnt_size = result.shape[3]

output_result = np.ones((df_size,cnt_size*3))

scale = (512/(imsize/8-1))
#选取热度图最大值作为该关键点的坐标
#并乘上比例
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