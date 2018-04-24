import sys
import os
import pandas as pd

dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, '../util'))
import preprocess as pre



df_file_path = os.path.join(dirname , "../train/Annotations/train.csv")
train_pad_path = os.path.join(dirname , "../train_pad/")
input_train_df = pd.read_csv(df_file_path )
pd_df = pre.pad_images(input_train_df )
pre.write_with_category(pd_df , pre_path=train_pad_path ,is_train = True)


