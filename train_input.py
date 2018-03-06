"""
按不同类型读图片，
并生成为：
x (m, 宽*高*3)
y (m,关键点*3)
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import misc

import imageio
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
# import pillow



pre_path = "train_pad/"
intput_blouse_file =  "Annotations/train_blouse_coord.csv"
scale = 4
#x (m, 宽*高*3)
def set_x_one_hot(df , scale = 1 , folder =  'train/'):
    filepath_test = folder+df.loc[0,'image_id']
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

    
    return y_coord[:,l_m_index]

    return y_coord




def get_x_y(df_size=100,scale=1,path =  "train_pad/Annotations/train_blouse_coord.csv"):
    df = pd.read_csv(path)

    df=df[:df_size]
    

    x_train = set_x_one_hot(df, scale, "train_pad/")

    y_train = set_y_coord(df , scale)

    print(x_train.shape)
    return x_train,y_train

if __name__ == "__main__":
    data_blouse_coord = pd.read_csv(pre_path + intput_blouse_file)

    small_data_blouse_coord = data_blouse_coord.loc[:1,:] 
    x_train = set_x_one_hot(small_data_blouse_coord, scale, pre_path)

    y_train = set_y_coord(small_data_blouse_coord , scale)
    print(x_train.shape , y_train.shape)