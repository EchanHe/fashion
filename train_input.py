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



pre_path = "train_pad/"
intput_blouse_file =  "Annotations/train_blouse_coord.csv"

input_dict = {0:"Annotations/train_blouse_coord.csv" , 1:"Annotations/train_outwear_coord.csv" , 2 :"Annotations/train_trousers_coord.csv"}

scale = 4
#x (m, 宽*高*3)
def set_x_flat(df , scale = 1 , folder =  'train/'):
    filepath_test = folder+df.iloc[0,0]
    img = Image.open(filepath_test)
    np_img = np.array(img)
    img = img.resize((int(np_img.shape[1]/scale),int(np_img.shape[0]/scale)))
    x_all =np.expand_dims( np.array(img).reshape((-1)) , axis=0)
    size= df.shape[0]  
    
    for idx,row in df.iterrows():
        filepath_test =folder+row['image_id']
        img = Image.open(filepath_test)
        np_img = np.array(img)
        
        img = img.resize((int(np_img.shape[1]/scale),int(np_img.shape[0]/scale)))
        np_img = np.array(img)
        np_img = np_img.reshape((-1))
        #print(np_img.shape)
#         np.concatenate(x_all,np.array(new_img))
        x_all = np.append(x_all,np.expand_dims(np_img,axis=0),axis=0)
#     print(x_all.shape)
#     x_all=x_all.reshape((size,-1))
#     print(x_all.shape)
#     np.savetxt('images.txt' , x_all)
    return x_all[1:]

def set_x_img(df , scale = 1 , folder =  'train/'):
    filepath_test = folder+df.iloc[0,0]
    img = Image.open(filepath_test)
    np_img = np.array(img)
    img = img.resize((int(np_img.shape[1]/scale),int(np_img.shape[0]/scale)))
    x_all =np.expand_dims( np.array(img) , axis=0)
    size= df.shape[0]  
    
    for idx,row in df.iterrows():
        filepath_test =folder+row['image_id']
        img = Image.open(filepath_test)
        np_img = np.array(img)
        img = img.resize((int(np_img.shape[1]/scale),int(np_img.shape[0]/scale)))

        np_img = np.array(img)
        #print(np_img.shape)
#         np.concatenate(x_all,np.array(new_img))
        x_all = np.append(x_all,np.expand_dims(np_img,axis=0),axis=0)
#     print(x_all.shape)
#     x_all=x_all.reshape((size,-1))
#     print(x_all.shape)
#     np.savetxt('images.txt' , x_all)
    return x_all[1:]
# y (m,关键点*3)
def set_y_coord(df,scale = 1):
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
    
  
    return_array = np.concatenate((y_coord[:,l_m_index],has_lm_data , is_vis_data),axis=1)
    return return_array
    #x1,y1 ... xn, yn
    #lm_1 ... lm_n
    #vis_1 ... vis_n




def get_x_y(df_size=-1,scale=1,pre_dir="train_pad/",cates=0,flat_x = True):

    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
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

def get_x_pred(df_size=100,scale=1,pre_dir="test_pad/",cates=0,flat_x = True):

    path = pre_dir +"Annotations/train_"+categories.get_cate_name(cates)+"_coord.csv"
    print("Read data from files: ",path)
    df = pd.read_csv(path)

    df=df[:df_size]
    
    if flat_x:
        x_train = set_x_flat(df, scale, "train_pad/")
    else:
        x_train = set_x_img(df, scale, "train_pad/")
    y_train = set_y_coord(df , scale)

    print("X shape: ",x_train.shape , "Y shape: " , y_train.shape)
    return x_train,y_train, df[["image_id","image_category"]]


if __name__ == "__main__":
    x_input,y_input = get_x_y(10,1)
    data_cols = y_input.shape[1]
    lm_cnt = int(y_input.shape[1]/4)
    id_coords = np.arange(0, lm_cnt*2)
    id_islm = np.arange(lm_cnt*2, lm_cnt*3)
    id_vis = np.arange(lm_cnt*3, lm_cnt*4)

    print(y_input[:,id_islm] )