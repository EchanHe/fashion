from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
import tensorflow as tf

slim = tf.contrib.slim


import numpy as np
import pandas as pd

import sys
import os
import argparse


util_folder_name = 'util'
model_folder_name = 'model'
image_folder_name = 'train_pad/'
valid_image_folder_name = 'valid_pad/'

dirname = os.path.dirname(__file__)
#根目录变量 rootdir
# rootdir = os.path.abspath(dirname)
rootdir = os.path.abspath(os.path.join(dirname,".."))
sys.path.append(os.path.join(rootdir, util_folder_name))
sys.path.append(os.path.join(rootdir, model_folder_name))
import categories
import train_input
import cpm

#命令行参数设置

parser = argparse.ArgumentParser()
parser.add_argument("cate", choices=["blouse" ,"outwear","trousers","skirt","dress" ],
                    help="The clothes category")
parser.add_argument("--imsize", default = "512", choices=["128","256","512" ], 
                    help="Image size for training")
parser.add_argument("total_size", type = int, 
                    help="Training set size")
parser.add_argument("-l", "--learning_rate", type=float, default =1e-04,        
                    help="learning rate")
parser.add_argument("-l2","--lambda_l2", type=float, default = 0.001  ,
                    help="lambda for L2 regularization")
parser.add_argument("-k", "--keep_prob", type=float, default = 0.5  ,             
                    help="keep probability for drop out")


def accuracy(result, coord, imsize):
    """测试准确率的函数

    """
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
    """
    训练模型的配置类
    """
    
    def __init__(self,category,imsize , learning_rate, lambda_l2=0.0 , keep_prob = 1.0):
        self.pre_path = os.path.join(rootdir , "./")
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
        self.params_dir = os.path.abspath(os.path.join(self.pre_path+"/params/CPM/" , category , folder)) +"/"
        self.load_filename = "cpm_"+category + ".ckpt-100"
    # =================== modify parameters ==================
    batch_size = 5
    initialize = True # True: train from scratch (should also
                    # delete the corresponding params files
                    # in params_dir);
                     # False: restore from pretrained model
    
    steps = "30000"   # if 'initialize = False', set steps to 
                     # where you want to restore
    # iterations config
    max_iteration = 101
    checkpoint_iters = 50
    summary_iters = 10
    validate_iters = 100
    
    

    gpu = '/gpu:0'


    
    # image config
    points_num = 7

#命令行参数赋值 配置类
args = parser.parse_args()


l = args.learning_rate
lambda_l2 = args.lambda_l2
keep_prob= args.keep_prob
category_name = args.cate
imsize = int(args.imsize)
total_size = int(args.total_size)


config = Config(category_name,imsize,l,lambda_l2,keep_prob)
config.points_num = categories.get_cate_lm_cnts(category_name)


img_path = os.path.join(rootdir ,image_folder_name)

file_name = os.path.join(rootdir , "train_pad/Annotations/train_"+category_name+"_coord.csv")

df = pd.read_csv(file_name)
if total_size>-1:
    df = df[:total_size]
input_data = train_input.data_stream(df,config.batch_size,is_train=True , pre_path = img_path )

##validation Set
##要改validdation 的文件
valid_file_name = os.path.join(rootdir , "train_pad/Annotations/train_"+category_name+"_coord.csv")
df_valid = pd.read_csv(valid_file_name)
valid_data = train_input.data_stream(df_valid,config.batch_size,is_train=True , pre_path = img_path)

print("Arguments: " + str(args))
print("steps: {}\nlearning rates: {}".format(config.max_iteration ,config.learning_rate))
print("read train data from: "+os.path.abspath(file_name))
print("read validation data from: "+os.path.abspath(valid_file_name))
print("Images in folder: "+ img_path)
print("==RETRAIN ?: {}==\n\tCheckpoint file from: {}".format(not config.initialize,config.load_filename))
print("========Leanring Rate: {} lambda: {} drop: {}========".format(config.learning_rate,config.lambda_l2 , config.keep_prob))


###模型####

model = cpm.CPM(config)
predict = model.inference_pose_vgg_l2()

loss = model.loss()
train_op = model.train_op(loss, model.global_step)

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

    merged = tf.summary.merge_all()
    logdir = config.logdir

    writer = tf.summary.FileWriter(logdir, sess.graph)

    valid_loss_ph = tf.placeholder(tf.float32, shape=(), name="valid_loss")
    loss_valid = tf.summary.scalar('loss_valid',valid_loss_ph ) 
    valid_acc_ph = tf.placeholder(tf.float32, shape=(), name="valid_acc")
    acc_valid = tf.summary.scalar('acc_valid',valid_acc_ph ) 

    for i in range(config.max_iteration):
        #训练阶段
        #以batch_size随机选择数据 迭代max_iteration次 
        X_train_mini,y_train_mini,coords_mini,vis_mini,center_mini,_ = input_data.get_next_batch()
        feed_dict = {
                    model.images: X_train_mini,
                    model.labels: y_train_mini,
                    model.vis_mask: vis_mini,
                    model.keep_prob:config.keep_prob
                    }
        sess.run(train_op, feed_dict=feed_dict)
        if (i+1) % config.summary_iters == 0:
            print("{} steps Loss: {}".format(i+1,sess.run(loss, feed_dict=feed_dict)))

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
            # Validation阶段 
            # 输出Validation set的准确率
            acc_list = np.array([])
            loss_list = np.array([])
            for i_df_test in np.arange(0,df_test.shape[0],config.batch_size):
                X_mini,y_mini,coords_mini,vis_mini,center_mini , _ = valid_data.get_next_batch_no_random()
                feed_dict = {
                        model.images: X_mini,
                        model.labels: y_mini,
                        model.vis_mask: vis_mini,
                        model.keep_prob: 1.0
                        }               
                _loss = sess.run(loss, feed_dict=feed_dict)

                result_mini = sess.run(predict, feed_dict=feed_dict)
                acc = accuracy(result_mini,coords_mini,imsize)
                
                acc_list = np.concatenate((acc_list,acc))
                loss_list = np.append(loss_list,_loss)

            print("\t VALIDATION {} steps: average acc and loss : {}  {}".format(i+1,np.mean(acc_list),np.mean(loss_list)))
            writer.add_summary(sess.run(loss_valid, feed_dict={valid_loss_ph: np.mean(loss_list)}) , tmp_global_step)  
            writer.add_summary(sess.run(acc_valid, feed_dict={valid_acc_ph: np.mean(acc_list)}) , tmp_global_step) 
        
        if (i + 1) % config.checkpoint_iters == 0:
            #写入checkpoint
            tmp_global_step = model.global_step.eval()
            model.save(sess, saver, config.save_filename,tmp_global_step)
    
    
    


