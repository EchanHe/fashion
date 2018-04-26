"""
按不同类型读图片，
并生成为：
x (m, 宽*高*3)
y (m,关键点*3)
"""

import numpy as np
import pandas as pd


from PIL import Image

from scipy.ndimage import gaussian_filter
# import cv2


def set_x_img(df , scale = 1 , folder =  'train/'):
    """
    返回[m,width,height,3]的数组
    """
    filepath_test = folder+df.iloc[0,0]
    img = Image.open(filepath_test)
    # np_img = np.array(img)
    # img = img.resize((int(np_img.shape[1]/scale),int(np_img.shape[0]/scale)))
    # x_all =np.expand_dims( np.array(img) , axis=0)
    size= df.shape[0]  
    width = int(img.size[0]/scale)
    x_all = np.zeros((size,width,width,3))

    i=0
    for idx,row in df.iterrows():
        filepath_test =folder+row['image_id']
        img = Image.open(filepath_test)
        img = img.resize((width,width))

        np_img = np.array(img)
        x_all[i,:,:,:] = np_img
        i+=1
        x_all
#     print(x_all.shape)
#     x_all=x_all.reshape((size,-1))
#     print(x_all.shape)
#     np.savetxt('images.txt' , x_all)
    return x_all
# y (m,关键点*3)
def set_y_coord(df,scale = 1,coord_only = False):
    """
    返回[m,2*landmark]的关键点坐标数组
    """
    columns = df.columns
    if "height" in columns or "width" in columns:
        l_m_columns = columns.drop(['image_id' , 'image_category','height','width'])
    else:
        l_m_columns = columns.drop(['image_id' , 'image_category'])
    y_coord = df[l_m_columns].as_matrix()

    y_coord[:,np.arange(0,y_coord.shape[1],3)] = y_coord[:,np.arange(0,y_coord.shape[1],3)]/scale
    y_coord[:,np.arange(1,y_coord.shape[1],3)] = y_coord[:,np.arange(1,y_coord.shape[1],3)]/scale
    
    l_m_index = np.append(np.arange(0,y_coord.shape[1],3), np.arange(1,y_coord.shape[1],3) )

    l_m_index = np.sort(l_m_index)
    vis_index = np.arange(2,y_coord.shape[1],3)

    
    # Whether has landmark point
    has_lm_data = y_coord[:,vis_index]
    has_lm_data[has_lm_data==0]=1
    has_lm_data[has_lm_data==-1]=0

    # Whether is visible
    is_vis_data = y_coord[:,vis_index]
    is_vis_data[np.logical_or( is_vis_data ==-1 , is_vis_data==0 )]=0
    
    if coord_only:
        return y_coord[:,l_m_index]
    return_array = np.concatenate((y_coord[:,l_m_index],has_lm_data , is_vis_data),axis=1)
    return return_array
    #x1,y1 ... xn, yn
    #lm_1 ... lm_n
    #vis_1 ... vis_n


def set_y_map(df,scale = 1):
    """
    返回[m,64,64,landmark]的关键点热点图
    """
    columns = df.columns
    if "height" in columns or "width" in columns:
        l_m_columns = columns.drop(['image_id' , 'image_category','height','width'])
    else:
        l_m_columns = columns.drop(['image_id' , 'image_category'])
    y_coord = df[l_m_columns].as_matrix()
    lm_cnt = int(y_coord.shape[1]/3)
    df_size = y_coord.shape[0]
    size = int(512/(scale*8))
    real_scale = 512/size
    y_map = np.zeros((df_size,size,size,lm_cnt))

    for j in range(df_size):
        for i in range(lm_cnt):
            x = int(round(y_coord[j,i*3]/real_scale))
            y = int(round(y_coord[j,i*3+1]/real_scale))
            is_lm = y_coord[j,i*3+2]
            

            if is_lm!=-1:
                if x>=size:
                    x=size-1
                if y>=size:
                    y=size-1
                y_map[j,y,x,i] = 20
                y_map[j,:,:,i] = gaussian_filter(y_map[j,:,:,i],sigma=2)
                # y_map[j,y,x,i]=1
    y_map = np.round(y_map,4)
    return y_map

def set_y_vis_map(df,scale = 1):
    """
    返回[m,64,64,landmark]的关键点是否能看到热点图
    """
    columns = df.columns
    if "height" in columns or "width" in columns:
        l_m_columns = columns.drop(['image_id' , 'image_category','height','width'])
    else:
        l_m_columns = columns.drop(['image_id' , 'image_category'])
    y_coord = df[l_m_columns].as_matrix()
    lm_cnt = int(y_coord.shape[1]/3)
    df_size = y_coord.shape[0]
    size = int(512/(scale*8))
    real_scale = 512/size
    y_map = np.ones((df_size,size,size,lm_cnt))

    for j in range(df_size):
        for i in range(lm_cnt):

            is_lm = y_coord[j,i*3+2]

            if is_lm==-1:
                y_map[j,:,:,i] = np.zeros((size,size))
                # y_map[j,y,x,i]=1
    y_map = np.round(y_map,4)
    return y_map


def set_y_center_map(df,scale = 1 , network_scale = 8):
    columns = df.columns
    if "height" in columns or "width" in columns:
        l_m_columns = columns.drop(['image_id' , 'image_category','height','width'])
    else:
        l_m_columns = columns.drop(['image_id' , 'image_category'])
    y_coord = df[l_m_columns].as_matrix()
    lm_cnt = int(y_coord.shape[1]/3)
    df_size = y_coord.shape[0]
    size = int(512/(scale*network_scale))
    real_scale = 512/size
    y_map = np.zeros((df_size,size,size,1))

    x_index = np.arange(0,lm_cnt*3 ,3)
    y_index = np.arange(1,lm_cnt*3 ,3)



    for j in range(df_size):
        x_id_array = y_coord[j,x_index]
        y_id_array = y_coord[j,y_index]

        x_mean = np.mean(x_id_array[x_id_array!=-1])
        y_mean = np.mean(y_id_array[y_id_array!=-1])
        x_mean = int(x_mean/real_scale)
        y_mean = int(y_mean/real_scale)

        y_map[j,y_mean,x_mean,0] = 200
        y_map[j,:,:,0] = gaussian_filter(y_map[j,:,:,0],sigma=4)
    y_map = np.round(y_map,4)
    return y_map




        
class data_stream:
    """
    读数据的类。
    一次读入所有图片的索引表
    然后随机或者按顺序返回batch大小的数据
    """
    def __init__(self,df,batch_size,is_train,scale=1,pre_path = "./train_pad/"):

        self.df  =df# "train_pad/Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
        self.pre_path = pre_path
        self.scale = scale
        # self.X = X
        self.df_size = df.shape[0]
        self.batch_size = batch_size
        self.is_train = is_train


        self.start_idx =0
        self.indices = np.arange(self.df_size)
        np.random.shuffle(self.indices)

        print("Init data class...")
        print("\tData shape: {}\n\tbatch_size:{}"\
            .format(self.df_size, self.batch_size))


    
    def get_next_batch(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        
        if self.start_idx >= (df_size - batch_size+1):
            self.start_idx = 0 
            self.indices = np.arange(self.df_size)
            np.random.shuffle(self.indices)

        # print(self.start_idx , self.start_idx+batch_size)
        # print(self.indices)
        excerpt = self.indices[self.start_idx:self.start_idx + batch_size]
        df_mini = self.df.iloc[excerpt]
        # print(excerpt)
        # print(df_mini.image_id)

        x_mini = set_x_img(df_mini, self.scale, self.pre_path)
        if is_train:
            y_mini = set_y_map(df_mini , self.scale)
            coords_mini = set_y_coord(df_mini , 1 , True)
            center_mini = set_y_center_map(df_mini , self.scale , 1)

            center_label_mini = set_y_center_map(df_mini , self.scale)
            vis_mini = set_y_vis_map(df_mini)
        # print("X shape: ",x_mini.shape , "Y shape: " , y_mini.shape)
        # print("Coords shape: ",coords_mini.shape , "Map shape: " , center_mini.shape)

        self.start_idx += batch_size

        if is_train:
            return x_mini , y_mini, coords_mini,vis_mini,center_mini,center_label_mini
        else:
            return x_mini

    def get_next_batch_no_random(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        
        if self.start_idx >= (df_size - batch_size+1):
            self.start_idx = 0 
        df_mini = self.df.iloc[self.start_idx : self.start_idx+batch_size]

        # print(df_mini.image_id)
        x_mini = set_x_img(df_mini, self.scale, self.pre_path)
        if is_train:
            y_mini = set_y_map(df_mini , self.scale)
            coords_mini = set_y_coord(df_mini , 1 , True)
            center_mini = set_y_center_map(df_mini , self.scale , 1)
            center_label_mini = set_y_center_map(df_mini , self.scale)
            vis_mini = set_y_vis_map(df_mini)
        self.start_idx += batch_size

        if is_train:
            return x_mini , y_mini, coords_mini,vis_mini,center_mini,center_label_mini
        else:
            return x_mini
        

    def get_next_batch_no_random_all(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        
        if self.start_idx >= (df_size - batch_size+1):
            df_mini = self.df.iloc[self.start_idx : ]
        else: 
            df_mini = self.df.iloc[self.start_idx : self.start_idx+batch_size]

        # print(df_mini.image_id)
        x_mini = set_x_img(df_mini, self.scale, self.pre_path)
        if is_train:
            y_mini = set_y_map(df_mini , self.scale)
            coords_mini = set_y_coord(df_mini , 1 , True)
            center_mini = set_y_center_map(df_mini , self.scale , 1)
            center_label_mini = set_y_center_map(df_mini , self.scale)
            vis_mini = set_y_vis_map(df_mini)
        self.start_idx += batch_size
        # print(x_mini.shape)
        if is_train:
            return x_mini , y_mini, coords_mini,vis_mini,center_mini,center_label_mini
        else:
            return x_mini

if __name__ == "__main__":
    category_name = "blouse"
    file_name = "./train_pad/Annotations/train_"+category_name+"_coord.csv"
    # ckpt_file_name = "/data/bop16yh/fashion/params/CPM/" + category_name +"/cpm_"+category_name + ".ckpt-20000"


    print("read data from: "+file_name)
    # print("====RETRAIN ?: {}==== \n  Checkpoint file from: {}".format(is_retrain,ckpt_file_name))
    df = pd.read_csv(file_name)

    input_data = data_stream(df,10,is_train=True)
    print(input_data.get_next_batch())
