"""
按不同类型读图片，
并生成为：
x (m, 宽*高*3)
y (m,关键点*3)
"""

import numpy as np
import pandas as pd


from PIL import Image
import categories

import time
from scipy.ndimage import gaussian_filter

pre_path = "train_pad/"
intput_blouse_file =  "Annotations/train_blouse_coord.csv"

input_dict = {0:"Annotations/train_blouse_coord.csv" , 1:"Annotations/train_outwear_coord.csv" , 2 :"Annotations/train_trousers_coord.csv"}

scale = 4
#x (m, 宽*高*3)
def set_x_flat(df , scale = 1 , folder =  'train/'):
    filepath_test = folder+df.iloc[0,0]
    img = Image.open(filepath_test)
    # np_img = np.array(img)
    # img = img.resize((int(np_img.shape[1]/scale),int(np_img.shape[0]/scale)))
    # x_all =np.expand_dims( np.array(img) , axis=0)
    size= df.shape[0]  
    width = int(img.size[0]/scale)
    x_all = np.zeros((size,width*width*3))
    i=0
    for idx,row in df.iterrows():
        filepath_test =folder+row['image_id']
        img = Image.open(filepath_test)
    
        
        img = img.resize((width,width))
        np_img = np.array(img)
        np_img = np_img.reshape((-1))
        #print(np_img.shape)
#         np.concatenate(x_all,np.array(new_img))
        x_all[i,:] = np_img
        i+=1
#     print(x_all.shape)
#     x_all=x_all.reshape((size,-1))
#     print(x_all.shape)
#     np.savetxt('images.txt' , x_all)
    return x_all

def set_x_img(df , scale = 1 , folder =  'train/'):
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
            x = int(y_coord[j,i*3]/real_scale)
            y = int(y_coord[j,i*3+1]/real_scale)
            # print(x,y)
            if (x>=0 and x <size) and (y>=0 and y <size):
                y_map[j,y,x,i] = 20
                y_map[j,:,:,i] = gaussian_filter(y_map[j,:,:,i],sigma=2)
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


def get_x_y(df_size=-1,scale=1,pre_dir="train_pad/",cates=0,flat_x = True):

    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord_augs.csv"
    print("Read data from files: ",path)
    df = pd.read_csv(path)
    if df_size !=-1:
        df=df[:df_size]
    
    if flat_x:
        x_train = set_x_flat(df, scale, "train_pad/")
    else:
        x_train = set_x_img(df, scale, "train_pad/")
    y_train = set_y_coord(df , scale)

    print("X shape: ",x_train.shape , "Y shape: " , y_train.shape)
    return x_train,y_train

def get_x_y_map(df_size=-1,scale=1,pre_dir="train_pad/",cates=0,flat_x = True):

    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
    print("Read data from files: ",path)
    df = pd.read_csv(path)
    if df_size !=-1:
        df=df[:df_size]
    
    if flat_x:
        x_train = set_x_flat(df, scale, "train_pad/")
    else:
        x_train = set_x_img(df, scale, "train_pad/")
    y_train = set_y_map(df , scale)
    y_coord = set_y_coord(df , 1 , True)

    center_map = set_y_center_map(df , scale , 1)

    print("X shape: ",x_train.shape , "Y shape: " , y_train.shape)
    print("Coords shape: ",y_coord.shape , "Map shape: " , center_map.shape)
    return x_train,y_train,y_coord , center_map

def get_x_y_center_map(df_size=-1,scale=1,pre_dir="train_pad/",cates=0,flat_x = True):

    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
    print("Read data from files: ",path)
    df = pd.read_csv(path)
    if df_size !=-1:
        df=df[:df_size]
    
    if flat_x:
        x_train = set_x_flat(df, scale, "train_pad/")
    else:
        x_train = set_x_img(df, scale, "train_pad/")
    y_train = set_y_center_map(df , scale)

    print("X shape: ",x_train.shape , "Y shape: " , y_train.shape)
    return x_train,y_train


def get_x_y_map_valid(df_size=-1,scale=1,pre_dir="./train_warm_up_pad/",cates=0,flat_x = True):

    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
    print("Read data from files: ",path)
    df = pd.read_csv(path)
    if df_size !=-1:
        df=df[:df_size]
    
    if flat_x:
        x_train = set_x_flat(df, scale, pre_dir)
    else:
        x_train = set_x_img(df, scale, pre_dir)
    y_train = set_y_map(df , scale)
    y_coord = set_y_coord(df , 1 , True)
    center_map = set_y_center_map(df , scale , 1)

    print("X shape: ",x_train.shape , "Y shape: " , y_train.shape)
    print("Coords shape: ",y_coord.shape , "Map shape: " , center_map.shape)
    return x_train,y_train    ,y_coord,center_map

def get_x_y_valid(df_size=-1,scale=1,pre_dir="./train_warm_up_pad/",cates=0,flat_x = True):

    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
    print("Read data from files: ",path)
    df = pd.read_csv(path)
    if df_size !=-1:
        df=df[:df_size]
    
    if flat_x:
        x_train = set_x_flat(df, scale, pre_dir)
    else:
        x_train = set_x_img(df, scale, pre_dir)
    y_train = set_y_coord(df , scale)

    print("X shape: ",x_train.shape , "Y shape: " , y_train.shape)
    return x_train,y_train

def get_x_y_time(df_size=-1,scale=1,pre_dir="train_pad/",cates=0,flat_x = True):

    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
    print("Read data from files: ",path)
    start_time = time.time()
    df = pd.read_csv(path)
    if df_size !=-1:
        df=df[:df_size]
    print("--- %s secs reading data ---" % ((time.time() - start_time)))
    if flat_x:
        x_train = set_x_flat(df, scale, "train_pad/")
    else:
        x_train = set_x_img(df, scale, "train_pad/")
    print("--- %s secs reading X ---" % ((time.time() - start_time)))
    y_train = set_y_coord(df , scale)
    print("--- %s secs reading Y ---" % ((time.time() - start_time)))
    print("X shape: ",x_train.shape , "Y shape: " , y_train.shape)
    return x_train,y_train   
 #
def get_x_y_s_e(start = 0,end=100,scale=1,pre_dir="train_pad/",cates=0,flat_x = True):
    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
    print("Read data from files: ",path)
    df = pd.read_csv(path)

    df=df[start:end]


    if flat_x:
        x_train = set_x_flat(df, scale, "train_pad/")
    else:
        x_train = set_x_img(df, scale, "train_pad/")

    y_train = set_y_coord(df , scale)

    print(x_train.shape)
    return x_train,y_train

def get_x_pred(df_size=100,scale=1,pre_dir="./test_pad/",cates=0,flat_x = True):

    path = pre_dir +"test_"+categories.get_cate_name(cates)+".csv"
    print("Read data from files: ",path)
    df = pd.read_csv(path)
    if df_size !=-1:
        df=df[:df_size]

    if flat_x:
        x_train = set_x_flat(df, scale, pre_dir)
    else:
        x_train = set_x_img(df, scale, pre_dir)

    print("X shape: ",x_train.shape )
    return x_train, df[["image_id","image_category"]]


if __name__ == "__main__":
    # x_input,y_input = get_x_y(10,1)
    # data_cols = y_input.shape[1]
    # lm_cnt = int(y_input.shape[1]/4)
    # id_coords = np.arange(0, lm_cnt*2)
    # id_islm = np.arange(lm_cnt*2, lm_cnt*3)
    # id_vis = np.arange(lm_cnt*3, lm_cnt*4)

    # print(y_input[:,id_islm] )


    x,y = get_x_y_time(200,scale=1)
    np.savetxt("foo.csv", x, delimiter=",")

    start_time = time.time()
    a=np.genfromtxt('foo.csv',delimiter=',')
    print("--- %s secs reading data ---" % ((time.time() - start_time)))
    # get_x_y_time(200,scale=1)
    # get_x_y_time(2000,scale=16)
    # get_x_y_time(2000 ,scale=16, flat_x = False)

class data:
    def __init__(self,X,Y,coords,batch_size,is_train):

        self.X = X
        self.df_size = X.shape[0]
        self.batch_size = batch_size
        self.is_train = is_train


        self.start_idx =0
        self.indices = np.arange(self.df_size)
        np.random.shuffle(self.indices)
        if is_train:
            self.coords = coords
            self.Y = Y

        print("Init data class...")
        print("\tX shape: {}\n\tY shape: {}\n\tbatch_size:{}"\
            .format(self.X.shape,self.Y.shape, self.batch_size))


    
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
        x_mini = self.X[excerpt]
        if is_train:
            y_mini = self.Y[excerpt] 
            if self.coords!=None:
                coords_mini = self.coords[start_idx:start_idx+batch_size]
            else:
                coords_mini = None
        self.start_idx += batch_size

        if is_train:
            return x_mini , y_mini, coords_mini
        else:
            return x_mini

    def get_next_batch_no_random(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        
        if self.start_idx >= (df_size - batch_size+1):
            self.start_idx = 0 
        x_mini = self.X[start_idx:start_idx+batch_size]
        if is_train:
            y_mini = self.Y[start_idx:start_idx+batch_size]
            if self.coords!=None:
                coords_mini = self.coords[start_idx:start_idx+batch_size]
            else:
                coords_mini = None
        self.start_idx += batch_size

        if is_train:
            return x_mini , y_mini, coords_mini
        else:
            return x_mini
        
class data2:
    def __init__(self,X,Y,coords,center,batch_size,is_train):

        self.X = X
        self.df_size = X.shape[0]
        self.batch_size = batch_size
        self.is_train = is_train


        self.start_idx =0
        self.indices = np.arange(self.df_size)
        np.random.shuffle(self.indices)
        if is_train:
            self.coords = coords
            self.Y = Y
            self.center = center

        print("Init data class...")
        print("\tX shape: {}\n\tY shape: {}\n\tbatch_size:{}"\
            .format(self.X.shape,self.Y.shape, self.batch_size))


    
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
        x_mini = self.X[excerpt]
        if is_train:
            y_mini = self.Y[excerpt] 
            center_mini = self.center[excerpt]
            coords_mini = self.coords[excerpt]

        self.start_idx += batch_size

        if is_train:
            return x_mini , y_mini, coords_mini,center_mini
        else:
            return x_mini

    def get_next_batch_no_random(self):
        batch_size = self.batch_size
        df_size = self.df_size
        is_train = self.is_train

        
        if self.start_idx >= (df_size - batch_size+1):
            self.start_idx = 0 
        x_mini = self.X[self.start_idx:self.start_idx+batch_size]
        if is_train:
            y_mini = self.Y[self.start_idx:self.start_idx+batch_size]
            coords_mini = self.coords[self.start_idx:self.start_idx+batch_size]
            center_mini= self.center[self.start_idx:self.start_idx+batch_size]
        self.start_idx += batch_size

        if is_train:
            return x_mini , y_mini, coords_mini,center_mini
        else:
            return x_mini
        