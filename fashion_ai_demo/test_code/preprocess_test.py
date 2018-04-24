import sys
import os
import pandas as pd

#调用~/util/preprocess
dirname = os.path.dirname(__file__)
sys.path.append(os.path.join(dirname, '../util'))
import preprocess as pre



df_file_path = os.path.join(dirname , "../test/test.csv")
test_pad_path = os.path.join(dirname , "../test_pad/")
input_test_df = pd.read_csv(df_file_path )
pd_df = pre.pad_images(input_test_df , is_train = False )
pre.write_with_category(pd_df , pre_path=test_pad_path ,is_train = False)