import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
# import pillow

#helper function:

def show_im_lms(df,index,scale=1 , pre_dir = 'train/'):
    #show landmarks
    columns = df.columns
    l_m_columns = columns.drop(['image_id' , 'image_category'])
    for col in l_m_columns:
        coord = df.loc[index,col]
        coord=coord.split('_')
        #change the string into integer
        coord = list(map(float, coord))

        if coord[0]!=-1:
            x=coord[0]/scale
            y=coord[1]/scale
            plt.plot(x,y,'*')
            
    filepath = pre_dir+df.loc[index,'image_id']
    img =  Image.open(filepath)
    width, height =img.size

    img = img.resize((width,height))
    plt.imshow(img)
    
def make_small_df(df , size =99):
    category_size = {}
    for idx,cate in enumerate(df.image_category.unique()):
        category_size[cate] = df.loc[df['image_category'] == cate,:].shape[0]
    
    df_result = df.loc[:size,:]
    beg=0
    for name,value in category_size.items():
        #print(name,value)
        beg+=value
        if beg>df.shape[0]:
            break
        df_result=pd.concat([df_result,df.loc[beg:beg+size,:]])
    return df_result.reset_index()

    return df
#输入原数据结构类型
#返回
def clean_columns(df , category):
    if category ==1:
        columns = ['image_id' , 'image_category','height','width',\
                   'neckline_left','neckline_right','shoulder_left' , 'shoulder_right','center_front',\
                   'armpit_left','armpit_right','top_hem_left','top_hem_right',\
                   'cuff_left_in', 'cuff_left_out', 'cuff_right_in','cuff_right_out']
        return df[columns]

def pad_img(img , size = 512):
    wid_diff = size-img.shape[0]
    height_diff = size - img.shape[1]
    left = int(wid_diff/2)
    right = size-left
    up = height_diff/2
    down = size-up
    img_pad = np.pad(img , ((left,right),(up,down),(0,0)) , 'constant')
    return img_pad