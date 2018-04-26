import numpy as np
import pandas as pd
import os
import categories

from PIL import Image
from PIL import ImageEnhance,ImageChops,ImageOps

def pad_img(np_img , size = 512):
    """ Pad 单张 Img.
    """
    wid_diff = size-np_img.shape[1]
    height_diff = size - np_img.shape[0]
    left = int(wid_diff/2)
    right = size-left-np_img.shape[1]
    up = int(height_diff/2)
    down = size-up- np_img.shape[0]
    
    img_pad = np.pad(np_img , ((up,down),(left,right),(0,0)) , 'constant',constant_values=0)
    
    return img_pad,left,up

def pad_images(df,im_size = 512  ,is_train=True, arg_path=""):
    """ 将size 小于 512 的图片 Pad 为 512 *512
    输入： 
        df: 图片的表格dataframe
        im_size : 图片应为大小， 默认512
        pre_path: 图片所存文件夹

    Padding后 图片存在[train|test]_pad 目录下
    索引表格存在 [train|test]_pad/Annotations/[train|test]_pad.csv

    """
    print("Padding images...")
    categories = df.image_category.unique()
    dirname = os.path.dirname(__file__)
    if is_train:
        if len(arg_path)>0:
            pre_path =  os.path.join(dirname , arg_path)
        else:
            pre_path =  os.path.join(dirname , "../train/")
        output_path = pre_path[:-1]+"_pad/"
        output_anno_path = os.path.join(output_path , "Annotations" )
        if not os.path.exists(output_anno_path):
            print("create folder: " + output_anno_path)
            os.makedirs(output_anno_path)
    else:
        test_offset_df = pd.DataFrame(columns=['image_id' , 'image_category','width','height'])
        pre_path = os.path.join(dirname , "../test/")
        output_path = pre_path[:-1]+"_pad/"
        output_anno_path = output_path
    print("input path: {}\noutput path:{}".format(os.path.abspath(pre_path),os.path.abspath(output_path)))
    for cate in categories:
        output_path_cate = os.path.join(output_path , "Images" , cate)
        
        if not os.path.exists(output_path_cate):
            print("create folder: " + output_path_cate)
            os.makedirs(output_path_cate)



    l_m_columns = df.columns.drop(['image_id' , 'image_category'])

    for idx,row in df.iterrows():
        filepath = pre_path+row['image_id']
        img = Image.open(filepath)

        if is_train ==False:
            test_offset_df=test_offset_df.append(pd.Series([row['image_id'],row['image_category'] ,
                                                img.size[0] ,img.size[1] ],
                                                index =test_offset_df.columns ) ,ignore_index=True)
        np_img = np.array(img)
        if np_img.shape[0] < im_size or np_img.shape[1] < im_size:

            (np_img,left,up) = pad_img(np_img , im_size)

            img = Image.fromarray(np_img, 'RGB')

            img.save(output_path+row['image_id'] )
            for col in l_m_columns:
                coord_list = row[col].split('_')
                coord_list = list(map(int,coord_list))
                
                if coord_list[0] != -1:
#                     print(coord_list)
                    #更新padding后的坐标
                    coord_list[0] +=left
                    coord_list[1] +=up
                    coord_list = list(map(str,coord_list))
                    coord_list = '_'.join(coord_list)
                    df.loc[idx,col] = coord_list
        else:
            img.save(output_path+row['image_id'] )
#                     print(coord_list)
    if is_train ==False:
        test_offset_df.to_csv(pre_path+"test_size.csv" , index=False)
        df.to_csv(os.path.join(output_anno_path , "test_pad.csv") , index = False)
    else:
        df.to_csv(os.path.join(output_anno_path , "train_pad.csv" ), index = False)
    return df


def write_with_category(df  , pre_path="./train_pad/" ,is_train = True):
    """将图片的表格分为5个类型的表格
        如果为训练集：调用 split_coord(df ,cate, output_path) 
            将x_y_vis 分为3；列并写入 train_<类型>_coord.csv中
    """
    print("Split into categories...")
    categories = df.image_category.unique()
    for cate in categories:
        df_cate = df.loc[df.image_category==cate,:].copy()
        if is_train:
            split_coord(df_cate , cate,pre_path)
        else:
            output_path = os.path.join(pre_path , "test_"+cate+".csv")
            df_cate.to_csv(output_path,index =False)

def split_coord(df ,cate, output_path):
    """将x_y_vis 分为3行
    """
    columns = df.columns
    l_m_columns = columns.drop(['image_id' , 'image_category'])

    cols = categories.get_columns(cate)
    for col in cols:
        coord_list = df[col].str.split('_')
        #if int(coord_list[0][1]) != -1:
        df[[col+"_x" , col+"_y", col+"_vis"]] = pd.DataFrame(coord_list.tolist(), index= df.index)
    df = df.drop(l_m_columns,axis=1)

    output_filename = "Annotations/train_"+cate+"_coord.csv"

    df.to_csv( output_path + output_filename, index = False)
    return df

# input_train_df = pd.read_csv("./train/Annotations/train.csv")
# pd_df = pad_images(input_train_df)
# write_with_category(pd_df , pre_path="./train_pad/" ,is_train = True)

# print( os.path.abspath(os.path.dirname(__file__)))
# input_test_df = pd.read_csv("./test/test.csv" )
# pd_df = pad_images(input_test_df , is_train = False)
# write_with_category(pd_df , pre_path="./test_pad/" ,is_train = False)