import numpy as np
import pandas as pd
import os,sys
from scipy import misc
 
from PIL import Image
# import pillow
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, '../util'))
import categories


"""
convert predication on test 


"""

all_columns = np.array( ['neckline_left', 'neckline_right',\
       'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left',\
       'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',\
       'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left',\
       'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left',\
       'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out',\
       'bottom_right_in', 'bottom_right_out'] )




input_dir = os.path.join(dirname , '../result/')
output_dir = os.path.join(dirname , '../result/')


df_size = pd.read_csv(os.path.join(dirname,"../test/test_size.csv"))

all_data = []
output_file = output_dir + "result_cpm.csv"
cates = ["blouse", "skirt","outwear","dress" ,"trousers"]
for cate in cates:
    
    input_file =input_dir+ "cpm_"+cate+"_512.csv"
    df_data_coord = pd.read_csv(input_file)
    output_data = df_data_coord.iloc[:,:2]
    output_data.columns = ['image_id', 'image_category']


    im_size=512

    #处理负数 以及处理Padding
    for idx,row in df_data_coord.iterrows():
        row_size = df_size.loc[df_size['image_id'] == row['image_id']]
        height = row_size.height.values[0]
        width = row_size.width.values[0]

        if height<im_size or width<im_size:
            wid_diff = im_size - width
            height_diff = im_size - height
            left = int(wid_diff/2)
            up = int(height_diff/2)
            x_start=2
            y_start = 3
            x_index = np.arange(x_start , row.shape[0] , 3)
            y_index= np.arange(y_start , row.shape[0] , 3)
            row[x_index] -=left
            row[y_index] -=up
            df_data_coord.loc[idx,:] = row
    begin_col=2
    interval=3
    cate_cols = categories.get_columns(cate)

    for col in all_columns:
        if col not in cate_cols:
            output_data[col] = '-1_-1_-1'
        else:
            coords = df_data_coord.iloc[:,begin_col:begin_col+interval].as_matrix().astype(int)
            vstr = np.vectorize(str)
            coords = vstr(coords)
            output_data[col] = ["_".join(item) for item in coords]
            begin_col+=interval
    all_data.append(output_data)   
output_all = pd.concat(all_data,ignore_index=True)
output_all.to_csv(output_file , index  = False)

    
