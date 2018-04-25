import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from PIL import Image
import os
# import pillow

#helper function:

def show_im_lms(df,index,scale=1 , pre_dir = 'train/'):
    #show landmarks
    fig  = plt.figure(figsize=(10,10))
    columns = df.columns
    l_m_columns = columns.drop(['image_id' , 'image_category'])
    markersize = 12
    for col in l_m_columns:
        coord = df.loc[index,col]
        coord=coord.split('_')
        #change the string into integer
        coord = list(map(float, coord))

        if coord[0]!=-1:
            x=coord[0]/scale
            y=coord[1]/scale
            plt.plot(x,y,'*',markersize = markersize)
            plt.text(x * (1 + 0.01), y * (1 + 0.01) , col, fontsize=12)
            
    filepath = pre_dir+df.loc[index,'image_id']
    img =  Image.open(filepath)
    width, height =img.size
    width = int(width/scale)
    height = int(height/scale)
    img = img.resize((width,height))
    plt.imshow(img)


def save_im_lms(df,index,scale=1 , pre_dir = 'train/'):
    #show landmarks
    fig  = plt.figure(figsize=(10,10))
    columns = df.columns
    l_m_columns = columns.drop(['image_id' , 'image_category'])
    markersize = 12
    for col in l_m_columns:
        coord = df.loc[index,col]
        coord=coord.split('_')
        #change the string into integer
        coord = list(map(float, coord))

        if coord[0]!=-1:
            x=coord[0]/scale
            y=coord[1]/scale
            plt.plot(x,y,'*',markersize = markersize)
            plt.text(x * (1 + 0.01), y * (1 + 0.01) , col, fontsize=12)
            
    filepath = pre_dir+df.loc[index,'image_id']
    img =  Image.open(filepath)
    width, height =img.size
    width = int(width/scale)
    height = int(height/scale)
    img = img.resize((width,height))
    plt.imshow(img)
    fig.savefig('foo.png')


def save_multi_im_lms(df , scale=1 , pre_dir = 'train/'):     
    columns = df.columns
    l_m_columns = columns.drop(['image_id' , 'image_category'])
    lm_cnt = int(l_m_columns.shape[0]/3)
    
    #show landmarks
    
    nrows = 2
    ncols =5
    markersize = 30
    #loop through image to show
    for i in np.arange(0,df.shape[0],10):
        print(i)
        fig  = plt.figure(figsize=(100,40))    
        df1 = df[i:i+10]
        for idx,row in df1.iterrows():
            
            plt.subplot(nrows,ncols,idx+1-i)
            for col in l_m_columns:
                coord = row[col]
                coord=coord.split('_')
                #change the string into integer
                coord = list(map(float, coord))

                if coord[0]!=-1:
                    x=coord[0]/scale
                    y=coord[1]/scale
                    plt.plot(x,y,'*',label=col,markersize=markersize)
                    plt.text(x * (1 + 0.01), y * (1 + 0.01) , col, fontsize=30,color='blue',bbox=dict(facecolor='red', alpha=0.2))
                    # print(x,y)

            filepath = pre_dir+row['image_id']
            plt.title(row['image_id']+"\n"+str(idx),fontsize=40)
            img =  Image.open(filepath)
            width, height =img.size
            width = int(width/scale)
            height = int(height/scale)
            img = img.resize((width,height))
            plt.imshow(img)
        if not os.path.exists(pre_dir+"result/"):
            os.makedirs(pre_dir+"result/")
        fig.savefig(pre_dir+"result/"+str(i)+".jpg")
        plt.close()

def show_im_coords_lms(df,index,scale=1 , pre_dir = 'train/'):
    #show landmarks
    fig  = plt.figure(figsize=(10,10))
    columns = df.columns
    l_m_columns = columns.drop(['image_id' , 'image_category'])
    lm_cnt = int(l_m_columns.shape[0]/3)
    markersize = 12
    for i in np.arange(lm_cnt):
        x = df.iloc[index , 2+i*3]
        y = df.iloc[index , 3+i*3]
        vis = df.iloc[index , 4+i*3]
        if vis!=-1:
            plt.plot(x,y,'*' , markersize=markersize)
            plt.text(x * (1 + 0.01), y * (1 + 0.01) , l_m_columns[i*3], fontsize=15,color='blue',bbox=dict(facecolor='red', alpha=0.2))
            
    filepath = pre_dir+df.loc[index,'image_id']
    img =  Image.open(filepath)
    width, height =img.size
    width = int(width/scale)
    height = int(height/scale)
    img = img.resize((width,height))
    plt.imshow(img)
    
    
def show_im_coords_lms_compare(df1 , df2, index_list ,scale=1 , pre_dir = 'train/'):     
    columns = df1.columns
    l_m_columns = columns.drop(['image_id' , 'image_category'])
    lm_cnt = int(l_m_columns.shape[0]/3)
    
    #show landmarks
    fig  = plt.figure(figsize=(20,10))    
    nrows = 2
    ncols = len(index_list)
    markersize = 12
    #loop through image to show
    for idx , index in enumerate(index_list):
        
        plt.subplot(nrows,ncols,idx+1)
        for i in np.arange(lm_cnt):
            x = df1.iloc[index , 2+i*3]
            y = df1.iloc[index , 3+i*3]
            vis = df1.iloc[index , 4+i*3]
            if vis!=-1:
                plt.plot(x,y,'*' , markersize=markersize )
#                 plt.text(x * (1 + 0.01), y * (1 + 0.01) , l_m_columns[i*3], fontsize=15,color='blue',bbox=dict(facecolor='red', alpha=0.2))

        filepath = pre_dir+df1.loc[index,'image_id']
        img =  Image.open(filepath)
        width, height =img.size
        width = int(width/scale)
        height = int(height/scale)
        img = img.resize((width,height))
        plt.imshow(img)


        plt.subplot(nrows,ncols,idx+1+ncols)

        for i in np.arange(lm_cnt):
            x = df2.iloc[index , 2+i*3]
            y = df2.iloc[index , 3+i*3]
            vis = df2.iloc[index , 4+i*3]
            if vis!=-1:
                plt.plot(x,y,'*' , markersize=markersize)
#                 plt.text(x * (1 + 0.01), y * (1 + 0.01) , l_m_columns[i*3], fontsize=10,color='blue',bbox=dict(facecolor='red', alpha=0.2))

        filepath = pre_dir+df2.loc[index,'image_id']
        img =  Image.open(filepath)
        width, height =img.size
        width = int(width/scale)
        height = int(height/scale)
        img = img.resize((width,height))
        plt.imshow(img)
        
        
# show image with origin     
def show_im_lms_compare(df1 , df2, index_list ,scale=1 , pre_dir = 'train/'):     
    columns = df1.columns
    l_m_columns = columns.drop(['image_id' , 'image_category'])
    lm_cnt = int(l_m_columns.shape[0]/3)
    
    #show landmarks
    fig  = plt.figure(figsize=(20,10))    
    nrows = 2
    ncols = len(index_list)
    markersize = 12
    #loop through image to show
    for idx , index in enumerate(index_list):
        
        plt.subplot(nrows,ncols,idx+1)
        for col in l_m_columns:
            coord = df1.loc[index,col]
            coord=coord.split('_')
            #change the string into integer
            coord = list(map(float, coord))

            if coord[0]!=-1:
                x=coord[0]/scale
                y=coord[1]/scale
                plt.plot(x,y,'*',label=col)
                plt.text(x * (1 + 0.01), y * (1 + 0.01) , col, fontsize=12)
                # print(x,y)

        filepath = pre_dir+df1.loc[index,'image_id']
        img =  Image.open(filepath)
        width, height =img.size
        width = int(width/scale)
        height = int(height/scale)
        img = img.resize((width,height))
        plt.imshow(img)


        plt.subplot(nrows,ncols,idx+1+ncols)

        for col in l_m_columns:
            coord = df2.loc[index,col]
            coord=coord.split('_')
            #change the string into integer
            coord = list(map(float, coord))

            if coord[0]!=-1:
                x=coord[0]/scale
                y=coord[1]/scale
                plt.plot(x,y,'*',label=col)
                plt.text(x * (1 + 0.01), y * (1 + 0.01) , col, fontsize=12)
                # print(x,y)

        filepath = pre_dir+df2.loc[index,'image_id']
        img =  Image.open(filepath)
        width, height =img.size
        width = int(width/scale)
        height = int(height/scale)
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