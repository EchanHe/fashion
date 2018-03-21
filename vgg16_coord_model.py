"""
VGG model setting.


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

output_kernels = 4096
coords_kernels = int(output_kernels/2)
bool_kernels = int(output_kernels/4)


def get_weight(imsize,lm_cnt):
    with tf.name_scope('fc'):
        weights = {
        # 1024 inputs, 10 outputs (class prediction)
        'out1': tf.Variable(tf.random_normal([512*int(imsize/32)*int(imsize/32), lm_cnt*2]) , name="w_1"),
        # 'out1': tf.Variable(tf.random_normal([512*int(imsize/32)*int(imsize/32), 4096]) , name="w_1"),
        'out2': tf.Variable(tf.random_normal([4096, lm_cnt*2]) , name="w_2")
        }


        biases= {
        'out1': tf.Variable(tf.random_normal([4096]) , name = "b_1"),
        'out2': tf.Variable(tf.random_normal([lm_cnt*2]) , name = "b_2")
        # 'out2': tf.Variable(tf.random_normal([num_classes]))
        }  
    return weights,biases 

def get_weight_all(imsize,lm_cnt):
    with tf.name_scope('fc'):
        weights = {
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.truncated_normal([512*int(imsize/32)*int(imsize/32), output_kernels]) , name="w_out"),
        # 'out1': tf.Variable(tf.random_normal([512*int(imsize/32)*int(imsize/32), 4096]) , name="w_1"),
        'out_coords': tf.Variable(tf.truncated_normal([coords_kernels, lm_cnt*2]) , name="w_coords"),
        'out_lm': tf.Variable(tf.truncated_normal([bool_kernels, lm_cnt]) , name="w_lm"),
        'out_vis': tf.Variable(tf.truncated_normal([bool_kernels, lm_cnt]) , name="w_vis")
        }


        biases= {
        'out': tf.Variable(tf.truncated_normal([4096]) , name = "b_out"),
        'out_coords': tf.Variable(tf.truncated_normal([lm_cnt*2]) , name="b_coords"),
        'out_lm': tf.Variable(tf.truncated_normal([lm_cnt]) , name="b_lm"),
        'out_vis': tf.Variable(tf.truncated_normal([lm_cnt]) , name="b_vis")
        # 'out2': tf.Variable(tf.random_normal([num_classes]))
        }  
    return weights,biases 

def get_weight_all_1_layer(imsize,lm_cnt):
    with tf.name_scope('fc'):
        shape = int((512*int(imsize/32)*int(imsize/32))/4)
        weights = {
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.truncated_normal([512*int(imsize/32)*int(imsize/32), output_kernels]) , name="w_out"),
        # 'out1': tf.Variable(tf.random_normal([512*int(imsize/32)*int(imsize/32), 4096]) , name="w_1"),
        'out_coords': tf.Variable(tf.truncated_normal([shape*2, lm_cnt*2]) , name="w_coords"),
        'out_lm': tf.Variable(tf.truncated_normal([shape, lm_cnt]) , name="w_lm"),
        'out_vis': tf.Variable(tf.truncated_normal([shape, lm_cnt]) , name="w_vis")
        }


        biases= {
        'out': tf.Variable(tf.truncated_normal([4096]) , name = "b_out"),
        'out_coords': tf.Variable(tf.truncated_normal([lm_cnt*2]) , name="b_coords"),
        'out_lm': tf.Variable(tf.truncated_normal([lm_cnt]) , name="b_lm"),
        'out_vis': tf.Variable(tf.truncated_normal([lm_cnt]) , name="b_vis")
        # 'out2': tf.Variable(tf.random_normal([num_classes]))
        }  
    return weights,biases 

def fc_layers(conv_layer,keep_prob,imsize,lm_cnt):

    weights,biases = get_weight(imsize,lm_cnt)
    coords_hat = tf.add(tf.matmul(conv_layer, weights['out1']), biases['out2'])
    coords_hat = tf.nn.dropout(coords_hat, keep_prob)

    return coords_hat

def fc_layers_all(conv_layer,keep_prob,imsize,lm_cnt):
    weights,biases = get_weight_all(imsize,lm_cnt)

    conv_layer = tf.add(tf.matmul(conv_layer, weights['out']), biases['out'])
    conv_layer = tf.nn.dropout(conv_layer, keep_prob)
    #conv_layer=[?, 4096]
    shape1 = int(4096/4)
    

    conv_layer_coord = conv_layer[:,:shape1*2]
    conv_layer_lm = conv_layer[:,shape1*2:shape1*3]
    conv_layer_vis = conv_layer[:,shape1*3:]
    print(conv_layer.shape,conv_layer_coord.shape,conv_layer_vis.shape)
    
    coords_hat = tf.add(tf.matmul(conv_layer_coord, weights['out_coords']), biases['out_coords'])
    logits_lm = tf.add(tf.matmul(conv_layer_lm, weights['out_lm']), biases['out_lm'])
    logits_vis = tf.add(tf.matmul(conv_layer_vis, weights['out_vis']), biases['out_vis'])

    return coords_hat, logits_lm , logits_vis
def fc_layers_all_1_layer(conv_layer,keep_prob,imsize,lm_cnt):
    weights,biases = get_weight_all_1_layer(imsize,lm_cnt)

    # conv_layer = tf.add(tf.matmul(conv_layer, weights['out']), biases['out'])
    # conv_layer = tf.nn.dropout(conv_layer, keep_prob)
    #conv_layer=[?, 4096]
    shape1 = int((512*int(imsize/32)*int(imsize/32))/4)
    

    conv_layer_coord = conv_layer[:,:shape1*2]
    conv_layer_lm = conv_layer[:,shape1*2:shape1*3]
    conv_layer_vis = conv_layer[:,shape1*3:]
    print(conv_layer.shape,conv_layer_coord.shape,conv_layer_vis.shape)
    
    coords_hat = tf.add(tf.matmul(conv_layer_coord, weights['out_coords']), biases['out_coords'])
    
    logits_lm = tf.add(tf.matmul(conv_layer_lm, weights['out_lm']), biases['out_lm'])

    logits_vis = tf.add(tf.matmul(conv_layer_vis, weights['out_vis']), biases['out_vis'])
    coords_hat = tf.nn.dropout(coords_hat, keep_prob)
    logits_lm = tf.nn.dropout(logits_lm, keep_prob)
    logits_vis = tf.nn.dropout(logits_vis, keep_prob)
    return coords_hat, logits_lm , logits_vis
def loss_all(coords_hat,logits_lm,logits_vis, Y):
    cnt = logits_lm.shape[1]
    loss_op_lm = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_lm, labels=Y[:,cnt*2:cnt*3])) 

    logits_vis_mask = tf.multiply(logits_vis,Y[:,cnt*2:cnt*3])
    GT_vis = tf.multiply(Y[:,cnt*3:],Y[:,cnt*2:cnt*3])
    # loss_op_vis = tf.reduce_mean(tf.square(logits_vis_mask - GT_vis)) / 2.

    loss_op_vis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_vis, labels=Y[:,cnt*3:])) 

    # tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    #     logits=GT_vis, labels=GT_vis))
    # loss_op_vis_1 = (tf.nn.softmax_cross_entropy_with_logits(
    #     logits=Y[:,cnt*2:cnt*3], labels=Y[:,cnt*2:cnt*3]))
    mask=tf.concat(
    [Y[:,cnt*2:cnt*3],Y[:,cnt*2:cnt*3]],
    axis=1)
    coord_diff =tf.subtract(coords_hat ,Y[:,:cnt*2])
    coord_diff = tf.multiply(coord_diff,mask)
    # coord_diff[:,:cnt] = tf.multiply(coord_diff[:,:cnt] ,Y[:,cnt*2:cnt*3] )
    # coord_diff[:,cnt:] = tf.multiply(coord_diff[:,cnt:] ,Y[:,cnt*2:cnt*3] )

    loss_op_coords = tf.reduce_mean(tf.square(coord_diff)) / 2.
    loss_op = loss_op_coords+loss_op_lm+loss_op_vis


    return loss_op,loss_op_lm,loss_op_vis,loss_op_coords


def acc_all(coords_hat,logits_lm,logits_vis, Y):
    lm_cnt = logits_lm.shape[1]

    pred_lm= tf.nn.sigmoid(logits_lm)
    pred_vis=  tf.nn.sigmoid(logits_vis)

    accuracy_lm =tf.reduce_mean(tf.cast(tf.equal(pred_lm, Y[:,lm_cnt*2:lm_cnt*3]), tf.float32))
    accuracy_vis =tf.reduce_mean(tf.cast(tf.equal(pred_vis, Y[:,lm_cnt*3:]), tf.float32))
    accuracy_bool =( accuracy_lm+accuracy_vis) /2.

    # accuracy_bool = accuracy_lm

    accuracy = tf.reduce_mean(tf.sqrt(tf.square(coords_hat - Y[:,:lm_cnt*2]))) 
    return accuracy ,accuracy_bool   

def loss(Y_hats, Y):
    # weights,biases = get_weight(imsize,lm_cnt)
    # l2_loss = tf.reduce_mean(0.01*tf.nn.l2_loss(weights['out2']) )
    loss_op = tf.reduce_mean(tf.square(tf.subtract(Y_hats ,Y))) / 2.
    return loss_op




















# imsize = 64
# learning_rate = 0.01
# ##steps and batch
# category_name = "skirt"

# total_size = 7000
# batch_size = 10
# num_steps =200
# display_step = 5
# dropout = 0.75

# save_log=False
# load_var = True
# ###Deal with input image
# # x_input,y_input = train_input.get_x_y(500,512/imsize, flat_x = False)
# x_input,y_input = train_input.get_x_y(df_size=total_size,scale=512/imsize,pre_dir="train_pad/",cates=category_name,flat_x = False)
# print("##############")
# print("Image Size: {} \n\nCategory: {}\ntotal_size: {}\nbatch_size: {}\nsteps: {}".format(imsize,category_name,total_size , batch_size,num_steps))
# print("##############")
# print("input X shape" ,  x_input.shape)


# data_cols = y_input.shape[1]
# lm_cnt =int( y_input.shape[1]/4)

# id_coords = np.arange(0, lm_cnt*2)
# id_islm = np.arange(lm_cnt*2, lm_cnt*3)
# id_vis = np.arange(lm_cnt*3, lm_cnt*4)
# y_lm = y_input[:,id_islm]
# y_vis = y_input[:,id_vis]

# y_coord = y_input[:,id_coords]

# X_train, X_test, y_train, y_test = sk.train_test_split(x_input,y_coord,test_size=0.1, random_state = 42)


# logs_path="./tmp/"
# model_path="./vgg_16.ckpt"
# output_model_path="./tmp/ckpt/vgg/vgg_16_out.ckpt"
# X = tf.placeholder(tf.float32, [None, imsize,imsize,3])


# net = vgg.vgg_16_with_img_size (X,
#       num_classes=1000,
#            is_training=True,
#            dropout_keep_prob=0.5,
#            spatial_squeeze=True,
#            scope='vgg_16',
#            fc_conv_padding='VALID',
#            global_pool=False,
#            im_size=imsize
#            )


# print("vgg net output shape: " , net.shape)






# vgg_fc_out = tf.placeholder(tf.float32, [None,int(imsize/32)*int(imsize/32) *512])
# Y = tf.placeholder(tf.float32, [None, lm_cnt*2])
# keep_prob = tf.placeholder(tf.float32)

# # coords_hat = tf.add(tf.matmul(vgg_fc_out, weights['out1']), biases['out1'])
# # coords_hat = tf.add(tf.matmul(coords_hat, weights['out2']), biases['out2'])


# coords_hat = tf.add(tf.matmul(vgg_fc_out, weights['out1']), biases['out2'])
# coords_hat = tf.nn.dropout(coords_hat, keep_prob)


# l2_loss = tf.reduce_mean(0.01*tf.nn.l2_loss(weights['out2']) )
# loss_op = tf.reduce_mean(tf.square(tf.subtract(coords_hat ,Y))) / 2. + l2_loss

# ph_learning_rate = tf.placeholder(tf.float32, shape=[])
# optimizer = tf.train.AdamOptimizer(learning_rate=ph_learning_rate)
# train_op = optimizer.minimize(loss_op)

# accuracy = tf.reduce_mean(tf.sqrt(tf.square(coords_hat - Y))) 

# init = tf.global_variables_initializer()

# #only save vars named with "vgg_16"
# variables_to_restore = slim.get_variables(scope="vgg_16")

# # for var in variables_to_restore:
# #     print(var)
# # for var in tf.trainable_variables():
# #     print (var.name , var.shape)
# restorer = tf.train.Saver(variables_to_restore)

# saver = tf.train.Saver()
# with tf.Session() as sess:

#     # Run the initializer
#     sess.run(init)
#     if save_log:
#         summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

#     if load_var:
#         print("Load model from path : " , model_path)
#         load_path = restorer.restore(sess, model_path) 
#     # results=sess.run(net, feed_dict={X:X_train})
    




#     for step in range(1, num_steps+1):

#         indices = np.arange(X_train.shape[0])
#         np.random.shuffle(indices)
#         # print(type(batch_x))
#         # print(batch_x.shape)
#         for start_idx in range(0, X_train.shape[0] - batch_size + 1, batch_size):
        
#             excerpt = indices[start_idx:start_idx + batch_size]
#             # Get pair of (X, y) of the current minibatch/chunk
#             X_train_mini = X_train[excerpt]
#             y_train_mini = y_train[excerpt]
#             results_mini=sess.run(net, feed_dict={X:X_train_mini})
#             sess.run(train_op, feed_dict={vgg_fc_out: results_mini, Y: y_train_mini,ph_learning_rate:learning_rate,keep_prob:dropout})

#         if step % display_step == 0 or step == 1:
#             # Calculate batch loss and accuracy
#             results_mini=sess.run(net, feed_dict={X:X_train_mini})
#             loss, acc,co = sess.run([loss_op, accuracy,coords_hat], feed_dict={vgg_fc_out: results_mini,
#                                                                  Y: y_train_mini,
#                                                                  keep_prob:1.0})
#             print("Step " + str(step) + ", Minibatch Loss= " + \
#                   "{:.4f}".format(loss) + ", Training Accuracy= " + \
#                   "{:.3f}".format(acc))
#             print("------")
#             print(np.sum(y_train_mini>=0))
#             print(np.sum(co>=0))
#             print(np.sum(co<0))
#         if step==60:
#             learning_rate=0.001
#         if step==100:
#             learning_rate=0.0001

#     print("Model save in path : " , output_model_path)            
#     save_path = saver.save(sess, output_model_path)   

#     results=sess.run(net, feed_dict={X:X_test})
#     print("Testing Accuracy:", \
#     sess.run(accuracy, feed_dict={vgg_fc_out: results,
#                                   Y: y_test,
#                                   keep_prob:1.0}))
#     a,acc = sess.run([coords_hat,accuracy], feed_dict={vgg_fc_out: results,
#                                       Y: y_test,
#                                       keep_prob:1.0})

#     np.savetxt("./vgg_result.csv", a,fmt='%i', delimiter=",")


