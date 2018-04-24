import tensorflow as tf
import tensorflow.contrib.layers as layers


class CPM:
    def __init__(self , config):
        """
        初始化模型的配置  
        """
                
        self.global_step = tf.get_variable("global_step", initializer=0,
                    dtype=tf.int32, trainable=False)
        self.start_learning_rate =config.learning_rate
        self.lambda_l2 = config.lambda_l2
        # self.stddev = config.stddev
        self.batch_size = config.batch_size
        # self.use_fp16 = config.use_fp16
        self.points_num = config.points_num
        self.params_dir = config.params_dir

        # self.fm_height = config.fm_height
        # self.fm_width = config.fm_width

        self.fm_height = int(config.img_height/8) #= int(config.img_height/8-1)
        self.fm_width = int(config.img_width/8 )#int(config.img_width/8-1)

        self.images = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, config.img_height, config.img_width, 3)
                )
        self.center_map = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, config.img_height, config.img_width, 1)
            )

        self.labels = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.fm_height, self.fm_width, self.points_num))
        self.coords = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.points_num * 2))
        self.vis_mask = tf.placeholder(
                dtype = tf.float32,
                shape = (self.batch_size, self.fm_height, self.fm_width, self.points_num))
        self.pred_images = tf.placeholder(
                dtype = tf.float32,
                shape = (None, config.img_height, config.img_width, 3)
                )
        self.keep_prob = tf.placeholder(tf.float32)

      
    def loss(self):
        """
        损失函数  
        """
        return tf.add( tf.add_n(tf.get_collection('losses')) , tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)), name = "total_loss")

    
    def add_to_euclidean_loss(self, batch_size, predicts, labels, name):
        """
        将每一stage的最后一层加入损失函数 
        """

        flatten_vis = tf.reshape(self.vis_mask, [batch_size, -1])
        flatten_labels = tf.multiply( tf.reshape(labels, [batch_size, -1]) ,flatten_vis)
        flatten_predicts = tf.multiply(tf.reshape(predicts, [batch_size, -1]) , flatten_vis)
        # flatten_labels = tf.reshape(labels, [batch_size, -1])
        # flatten_predicts = tf.reshape(predicts, [batch_size, -1])
        # print(flatten_labels , flatten_predicts)
        with tf.name_scope(name) as scope:
            euclidean_loss = tf.sqrt(tf.reduce_sum(
              tf.square(tf.subtract(flatten_predicts, flatten_labels)), 1))
            # print(euclidean_loss)
            euclidean_loss_mean = tf.reduce_mean(euclidean_loss,
                name='euclidean_loss_mean')

        tf.add_to_collection("losses", euclidean_loss_mean)

    def train_op(self, total_loss, global_step):
        """
        Optimizer
        """
        self._loss_summary(total_loss)

        self.learning_rate = tf.train.exponential_decay(self.start_learning_rate, global_step,
                                                   5000, 0.5, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        grads = optimizer.compute_gradients(total_loss)

        apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)    

        return apply_gradient_op

    ###存储与恢复参数checkpoint####
    def save(self, sess, saver, filename, global_step):
        path = saver.save(sess, self.params_dir+filename, global_step=global_step)
        print ("Save params at " + path)

    def restore(self, sess, saver, filename):
        print ("Restore from previous model: ", self.params_dir+filename)
        saver.restore(sess, self.params_dir+filename)
    ###存储loss 和 中间的图片进入log###
    def _loss_summary(self, loss):
        tf.summary.scalar(loss.op.name + "_raw", loss)

    def _image_summary(self, x, channels):
        def sub(batch, idx):
            name = x.op.name
            tmp = x[batch, :, :, idx] * 255
            tmp = tf.expand_dims(tmp, axis = 2)
            tmp = tf.expand_dims(tmp, axis = 0)
            tf.summary.image(name + '-' + str(idx), tmp, max_outputs = 100)
        if (self.batch_size > 1):
          for idx in range(channels):
            # the first batch
            sub(0, idx)
            # the last batch
            sub(-1, idx)
        else:
          for idx in range(channels):
            sub(0, idx)

    def _fm_summary(self, predicts):
      with tf.name_scope("fcn_summary") as scope:
          self._image_summary(self.labels, self.points_num)
          tmp_predicts = tf.nn.relu(predicts)
          self._image_summary(tmp_predicts, self.points_num)


    def inference_pose_vgg(self, is_train=True):

        center_map = self.center_map
        lm_cnt = self.points_num
        if is_train:
            image = self.images
        else:
            image = self.pred_images
        with tf.variable_scope('PoseNet'):
            # pool_center_lower = layers.avg_pool2d(center_map, 9, 8, padding='SAME')
            conv1_1 = layers.conv2d(image, 64, 3, 1, activation_fn=None, scope='conv1_1')
            conv1_1 = tf.nn.relu(conv1_1)
            conv1_2 = layers.conv2d(conv1_1, 64, 3, 1, activation_fn=None, scope='conv1_2')
            conv1_2 = tf.nn.relu(conv1_2)
            pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
            conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1, activation_fn=None, scope='conv2_1')
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_2 = layers.conv2d(conv2_1, 128, 3, 1, activation_fn=None, scope='conv2_2')
            conv2_2 = tf.nn.relu(conv2_2)
            pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
            conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1, activation_fn=None, scope='conv3_1')
            conv3_1 = tf.nn.relu(conv3_1)
            conv3_2 = layers.conv2d(conv3_1, 256, 3, 1, activation_fn=None, scope='conv3_2')
            conv3_2 = tf.nn.relu(conv3_2)
            conv3_3 = layers.conv2d(conv3_2, 256, 3, 1, activation_fn=None, scope='conv3_3')
            conv3_3 = tf.nn.relu(conv3_3)
            conv3_4 = layers.conv2d(conv3_3, 256, 3, 1, activation_fn=None, scope='conv3_4')
            conv3_4 = tf.nn.relu(conv3_4)
            pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
            conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1, activation_fn=None, scope='conv4_1')
            conv4_1 = tf.nn.relu(conv4_1)
            conv4_2 = layers.conv2d(conv4_1, 512, 3, 1, activation_fn=None, scope='conv4_2')
            conv4_2 = tf.nn.relu(conv4_2)
            conv4_3_CPM = layers.conv2d(conv4_2, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
            conv4_3_CPM = tf.nn.relu(conv4_3_CPM)
            conv4_4_CPM = layers.conv2d(conv4_3_CPM, 256, 3, 1, activation_fn=None, scope='conv4_4_CPM')
            conv4_4_CPM = tf.nn.relu(conv4_4_CPM)
            conv4_5_CPM = layers.conv2d(conv4_4_CPM, 256, 3, 1, activation_fn=None, scope='conv4_5_CPM')
            conv4_5_CPM = tf.nn.relu(conv4_5_CPM)
            conv4_6_CPM = layers.conv2d(conv4_5_CPM, 256, 3, 1, activation_fn=None, scope='conv4_6_CPM')
            conv4_6_CPM = tf.nn.relu(conv4_6_CPM)
            conv4_7_CPM = layers.conv2d(conv4_6_CPM, 128, 3, 1, activation_fn=None, scope='conv4_7_CPM')
            conv4_7_CPM = tf.nn.relu(conv4_7_CPM)
            conv5_1_CPM = layers.conv2d(conv4_7_CPM, 512, 1, 1, activation_fn=None, scope='conv5_1_CPM')
            conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
            conv5_2_CPM = layers.conv2d(conv5_1_CPM, lm_cnt, 1, 1, activation_fn=None, scope='conv5_2_CPM')
            concat_stage2 = tf.concat(axis=3, values=[conv5_2_CPM, conv4_7_CPM])
            Mconv1_stage2 = layers.conv2d(concat_stage2, 128, 7, 1, activation_fn=None, scope='Mconv1_stage2')
            Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
            Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 128, 7, 1, activation_fn=None, scope='Mconv2_stage2')
            Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
            Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 128, 7, 1, activation_fn=None, scope='Mconv3_stage2')
            Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
            Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 128, 7, 1, activation_fn=None, scope='Mconv4_stage2')
            Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
            Mconv5_stage2 = layers.conv2d(Mconv4_stage2, 128, 7, 1, activation_fn=None, scope='Mconv5_stage2')
            Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
            Mconv6_stage2 = layers.conv2d(Mconv5_stage2, 128, 1, 1, activation_fn=None, scope='Mconv6_stage2')
            Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
            Mconv7_stage2 = layers.conv2d(Mconv6_stage2, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage2')
            concat_stage3 = tf.concat(axis=3, values=[Mconv7_stage2, conv4_7_CPM])
            Mconv1_stage3 = layers.conv2d(concat_stage3, 128, 7, 1, activation_fn=None, scope='Mconv1_stage3')
            Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
            Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 128, 7, 1, activation_fn=None, scope='Mconv2_stage3')
            Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
            Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 128, 7, 1, activation_fn=None, scope='Mconv3_stage3')
            Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
            Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 128, 7, 1, activation_fn=None, scope='Mconv4_stage3')
            Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
            Mconv5_stage3 = layers.conv2d(Mconv4_stage3, 128, 7, 1, activation_fn=None, scope='Mconv5_stage3')
            Mconv5_stage3 = tf.nn.relu(Mconv5_stage3)
            Mconv6_stage3 = layers.conv2d(Mconv5_stage3, 128, 1, 1, activation_fn=None, scope='Mconv6_stage3')
            Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
            Mconv7_stage3 = layers.conv2d(Mconv6_stage3, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage3')
            concat_stage4 = tf.concat(axis=3, values=[Mconv7_stage3, conv4_7_CPM])
            Mconv1_stage4 = layers.conv2d(concat_stage4, 128, 7, 1, activation_fn=None, scope='Mconv1_stage4')
            Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
            Mconv2_stage4 = layers.conv2d(Mconv1_stage4, 128, 7, 1, activation_fn=None, scope='Mconv2_stage4')
            Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
            Mconv3_stage4 = layers.conv2d(Mconv2_stage4, 128, 7, 1, activation_fn=None, scope='Mconv3_stage4')
            Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
            Mconv4_stage4 = layers.conv2d(Mconv3_stage4, 128, 7, 1, activation_fn=None, scope='Mconv4_stage4')
            Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
            Mconv5_stage4 = layers.conv2d(Mconv4_stage4, 128, 7, 1, activation_fn=None, scope='Mconv5_stage4')
            Mconv5_stage4 = tf.nn.relu(Mconv5_stage4)
            Mconv6_stage4 = layers.conv2d(Mconv5_stage4, 128, 1, 1, activation_fn=None, scope='Mconv6_stage4')
            Mconv6_stage4 = tf.nn.relu(Mconv6_stage4)
            Mconv7_stage4 = layers.conv2d(Mconv6_stage4, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage4')
            concat_stage5 = tf.concat(axis=3, values=[Mconv7_stage4, conv4_7_CPM])
            Mconv1_stage5 = layers.conv2d(concat_stage5, 128, 7, 1, activation_fn=None, scope='Mconv1_stage5')
            Mconv1_stage5 = tf.nn.relu(Mconv1_stage5)
            Mconv2_stage5 = layers.conv2d(Mconv1_stage5, 128, 7, 1, activation_fn=None, scope='Mconv2_stage5')
            Mconv2_stage5 = tf.nn.relu(Mconv2_stage5)
            Mconv3_stage5 = layers.conv2d(Mconv2_stage5, 128, 7, 1, activation_fn=None, scope='Mconv3_stage5')
            Mconv3_stage5 = tf.nn.relu(Mconv3_stage5)
            Mconv4_stage5 = layers.conv2d(Mconv3_stage5, 128, 7, 1, activation_fn=None, scope='Mconv4_stage5')
            Mconv4_stage5 = tf.nn.relu(Mconv4_stage5)
            Mconv5_stage5 = layers.conv2d(Mconv4_stage5, 128, 7, 1, activation_fn=None, scope='Mconv5_stage5')
            Mconv5_stage5 = tf.nn.relu(Mconv5_stage5)
            Mconv6_stage5 = layers.conv2d(Mconv5_stage5, 128, 1, 1, activation_fn=None, scope='Mconv6_stage5')
            Mconv6_stage5 = tf.nn.relu(Mconv6_stage5)
            Mconv7_stage5 = layers.conv2d(Mconv6_stage5, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage5')
            concat_stage6 = tf.concat(axis=3, values=[Mconv7_stage5, conv4_7_CPM])
            Mconv1_stage6 = layers.conv2d(concat_stage6, 128, 7, 1, activation_fn=None, scope='Mconv1_stage6')
            Mconv1_stage6 = tf.nn.relu(Mconv1_stage6)
            Mconv2_stage6 = layers.conv2d(Mconv1_stage6, 128, 7, 1, activation_fn=None, scope='Mconv2_stage6')
            Mconv2_stage6 = tf.nn.relu(Mconv2_stage6)
            Mconv3_stage6 = layers.conv2d(Mconv2_stage6, 128, 7, 1, activation_fn=None, scope='Mconv3_stage6')
            Mconv3_stage6 = tf.nn.relu(Mconv3_stage6)
            Mconv4_stage6 = layers.conv2d(Mconv3_stage6, 128, 7, 1, activation_fn=None, scope='Mconv4_stage6')
            Mconv4_stage6 = tf.nn.relu(Mconv4_stage6)
            Mconv5_stage6 = layers.conv2d(Mconv4_stage6, 128, 7, 1, activation_fn=None, scope='Mconv5_stage6')
            Mconv5_stage6 = tf.nn.relu(Mconv5_stage6)
            Mconv6_stage6 = layers.conv2d(Mconv5_stage6, 128, 1, 1, activation_fn=None, scope='Mconv6_stage6')
            Mconv6_stage6 = tf.nn.relu(Mconv6_stage6)
            Mconv7_stage6 = layers.conv2d(Mconv6_stage6, lm_cnt, 1, 1, activation_fn=None, scope='Mconv7_stage6')
            # print(conv5_2_CPM)
            # print(Mconv7_stage6)
            # print(self.labels)
            if is_train:
                self.add_to_euclidean_loss(self.batch_size, conv5_2_CPM, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage2, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage3, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage4, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage5, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage6, self.labels, 'st')
        # self.add_to_accuracy(Mconv7_stage6)
        self._fm_summary(Mconv7_stage6)
        return Mconv7_stage6





    def inference_pose_vgg_l2(self, is_train=True):

        center_map = self.center_map
        lm_cnt = self.points_num
        if is_train:
            image = self.images
        else:
            image = self.pred_images
        with tf.variable_scope('PoseNet'):
            print("lambda : {} keep prob: {} ".format(self.lambda_l2 , self.keep_prob))
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0005)
            # pool_center_lower = layers.avg_pool2d(center_map, 9, 8, padding='SAME')
            conv1_1 = layers.conv2d(image, 64, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv1_1')
            conv1_1 = tf.nn.relu(conv1_1)
            conv1_2 = layers.conv2d(conv1_1, 64, 3, 1, weights_regularizer=regularizer,activation_fn=None, scope='conv1_2')
            conv1_2 = tf.nn.relu(conv1_2)
            # return conv1_2
            pool1_stage1 = layers.max_pool2d(conv1_2, 2, 2)
            conv2_1 = layers.conv2d(pool1_stage1, 128, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv2_1')
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_2 = layers.conv2d(conv2_1, 128, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv2_2')
            conv2_2 = tf.nn.relu(conv2_2)
            pool2_stage1 = layers.max_pool2d(conv2_2, 2, 2)
            conv3_1 = layers.conv2d(pool2_stage1, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv3_1')
            conv3_1 = tf.nn.relu(conv3_1)
            conv3_2 = layers.conv2d(conv3_1, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv3_2')
            conv3_2 = tf.nn.relu(conv3_2)
            conv3_3 = layers.conv2d(conv3_2, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv3_3')
            conv3_3 = tf.nn.relu(conv3_3)
            conv3_4 = layers.conv2d(conv3_3, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv3_4')
            conv3_4 = tf.nn.relu(conv3_4)
            pool3_stage1 = layers.max_pool2d(conv3_4, 2, 2)
            conv4_1 = layers.conv2d(pool3_stage1, 512, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_1')
            conv4_1 = tf.nn.relu(conv4_1)
            conv4_2 = layers.conv2d(conv4_1, 512, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_2')
            conv4_2 = tf.nn.relu(conv4_2)
            conv4_3_CPM = layers.conv2d(conv4_2, 256, 3, 1, activation_fn=None, scope='conv4_3_CPM')
            conv4_3_CPM = tf.nn.relu(conv4_3_CPM)
            conv4_4_CPM = layers.conv2d(conv4_3_CPM, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_4_CPM')
            conv4_4_CPM = tf.nn.relu(conv4_4_CPM)
            conv4_5_CPM = layers.conv2d(conv4_4_CPM, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_5_CPM')
            conv4_5_CPM = tf.nn.relu(conv4_5_CPM)
            conv4_6_CPM = layers.conv2d(conv4_5_CPM, 256, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_6_CPM')
            conv4_6_CPM = tf.nn.relu(conv4_6_CPM)
            conv4_7_CPM = layers.conv2d(conv4_6_CPM, 128, 3, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv4_7_CPM')
            conv4_7_CPM = tf.nn.relu(conv4_7_CPM)
            conv5_1_CPM = layers.conv2d(conv4_7_CPM, 512, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv5_1_CPM')
            conv5_1_CPM = tf.nn.relu(conv5_1_CPM)
            conv5_1_CPM = tf.nn.dropout(conv5_1_CPM, self.keep_prob)
            conv5_2_CPM = layers.conv2d(conv5_1_CPM, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='conv5_2_CPM')
            concat_stage2 = tf.concat(axis=3, values=[conv5_2_CPM, conv4_7_CPM])
            Mconv1_stage2 = layers.conv2d(concat_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage2')
            Mconv1_stage2 = tf.nn.relu(Mconv1_stage2)
            Mconv2_stage2 = layers.conv2d(Mconv1_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv2_stage2')
            Mconv2_stage2 = tf.nn.relu(Mconv2_stage2)
            Mconv3_stage2 = layers.conv2d(Mconv2_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage2')
            Mconv3_stage2 = tf.nn.relu(Mconv3_stage2)
            Mconv4_stage2 = layers.conv2d(Mconv3_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage2')
            Mconv4_stage2 = tf.nn.relu(Mconv4_stage2)
            Mconv5_stage2 = layers.conv2d(Mconv4_stage2, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage2')
            Mconv5_stage2 = tf.nn.relu(Mconv5_stage2)
            Mconv6_stage2 = layers.conv2d(Mconv5_stage2, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage2')
            Mconv6_stage2 = tf.nn.relu(Mconv6_stage2)
            Mconv6_stage2 = tf.nn.dropout(Mconv6_stage2, self.keep_prob)
            Mconv7_stage2 = layers.conv2d(Mconv6_stage2, lm_cnt, 1, 1, weights_regularizer=regularizer,activation_fn=None, scope='Mconv7_stage2')
            concat_stage3 = tf.concat(axis=3, values=[Mconv7_stage2, conv4_7_CPM])
            Mconv1_stage3 = layers.conv2d(concat_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage3')
            Mconv1_stage3 = tf.nn.relu(Mconv1_stage3)
            Mconv2_stage3 = layers.conv2d(Mconv1_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv2_stage3')
            Mconv2_stage3 = tf.nn.relu(Mconv2_stage3)
            Mconv3_stage3 = layers.conv2d(Mconv2_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage3')
            Mconv3_stage3 = tf.nn.relu(Mconv3_stage3)
            Mconv4_stage3 = layers.conv2d(Mconv3_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage3')
            Mconv4_stage3 = tf.nn.relu(Mconv4_stage3)
            Mconv5_stage3 = layers.conv2d(Mconv4_stage3, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage3')
            Mconv5_stage3 = tf.nn.relu(Mconv5_stage3)
            Mconv6_stage3 = layers.conv2d(Mconv5_stage3, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage3')
            Mconv6_stage3 = tf.nn.relu(Mconv6_stage3)
            Mconv6_stage3 = tf.nn.dropout(Mconv6_stage3, self.keep_prob)
            Mconv7_stage3 = layers.conv2d(Mconv6_stage3, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv7_stage3')
            concat_stage4 = tf.concat(axis=3, values=[Mconv7_stage3, conv4_7_CPM])
            Mconv1_stage4 = layers.conv2d(concat_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage4')
            Mconv1_stage4 = tf.nn.relu(Mconv1_stage4)
            Mconv2_stage4 = layers.conv2d(Mconv1_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv2_stage4')
            Mconv2_stage4 = tf.nn.relu(Mconv2_stage4)
            Mconv3_stage4 = layers.conv2d(Mconv2_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage4')
            Mconv3_stage4 = tf.nn.relu(Mconv3_stage4)
            Mconv4_stage4 = layers.conv2d(Mconv3_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage4')
            Mconv4_stage4 = tf.nn.relu(Mconv4_stage4)
            Mconv5_stage4 = layers.conv2d(Mconv4_stage4, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage4')
            Mconv5_stage4 = tf.nn.relu(Mconv5_stage4)
            Mconv6_stage4 = layers.conv2d(Mconv5_stage4, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage4')
            Mconv6_stage4 = tf.nn.relu(Mconv6_stage4)
            Mconv6_stage4 = tf.nn.dropout(Mconv6_stage4, self.keep_prob)
            Mconv7_stage4 = layers.conv2d(Mconv6_stage4, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv7_stage4')
            concat_stage5 = tf.concat(axis=3, values=[Mconv7_stage4, conv4_7_CPM])
            Mconv1_stage5 = layers.conv2d(concat_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage5')
            Mconv1_stage5 = tf.nn.relu(Mconv1_stage5)
            Mconv2_stage5 = layers.conv2d(Mconv1_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv2_stage5')
            Mconv2_stage5 = tf.nn.relu(Mconv2_stage5)
            Mconv3_stage5 = layers.conv2d(Mconv2_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage5')
            Mconv3_stage5 = tf.nn.relu(Mconv3_stage5)
            Mconv4_stage5 = layers.conv2d(Mconv3_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage5')
            Mconv4_stage5 = tf.nn.relu(Mconv4_stage5)
            Mconv5_stage5 = layers.conv2d(Mconv4_stage5, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage5')
            Mconv5_stage5 = tf.nn.relu(Mconv5_stage5)
            Mconv6_stage5 = layers.conv2d(Mconv5_stage5, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage5')
            Mconv6_stage5 = tf.nn.relu(Mconv6_stage5)
            Mconv6_stage5 = tf.nn.dropout(Mconv6_stage5, self.keep_prob)
            Mconv7_stage5 = layers.conv2d(Mconv6_stage5, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv7_stage5')
            concat_stage6 = tf.concat(axis=3, values=[Mconv7_stage5, conv4_7_CPM])
            Mconv1_stage6 = layers.conv2d(concat_stage6, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv1_stage6')
            Mconv1_stage6 = tf.nn.relu(Mconv1_stage6)
            Mconv2_stage6 = layers.conv2d(Mconv1_stage6, 128, 7, 1, weights_regularizer=regularizer,activation_fn=None, scope='Mconv2_stage6')
            Mconv2_stage6 = tf.nn.relu(Mconv2_stage6)
            Mconv3_stage6 = layers.conv2d(Mconv2_stage6, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv3_stage6')
            Mconv3_stage6 = tf.nn.relu(Mconv3_stage6)
            Mconv4_stage6 = layers.conv2d(Mconv3_stage6, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv4_stage6')
            Mconv4_stage6 = tf.nn.relu(Mconv4_stage6)
            Mconv5_stage6 = layers.conv2d(Mconv4_stage6, 128, 7, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv5_stage6')
            Mconv5_stage6 = tf.nn.relu(Mconv5_stage6)
            Mconv6_stage6 = layers.conv2d(Mconv5_stage6, 128, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv6_stage6')
            Mconv6_stage6 = tf.nn.relu(Mconv6_stage6)
            Mconv6_stage6 = tf.nn.dropout(Mconv6_stage6, self.keep_prob)
            Mconv7_stage6 = layers.conv2d(Mconv6_stage6, lm_cnt, 1, 1,weights_regularizer=regularizer, activation_fn=None, scope='Mconv7_stage6')
            # print(conv5_2_CPM)
            # print(Mconv7_stage6)
            # print(self.labels)
            if is_train:
                self.add_to_euclidean_loss(self.batch_size, conv5_2_CPM, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage2, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage3, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage4, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage5, self.labels, 'st')
                self.add_to_euclidean_loss(self.batch_size, Mconv7_stage6, self.labels, 'st')
        # self.add_to_accuracy(Mconv7_stage6)
        self._fm_summary(Mconv7_stage6)
        return Mconv7_stage6