import numpy as np 

def get_columns(i):

    cates_map = {"blouse":0 ,"outwear":1,"trousers":2,"skirt":3,"dress":4 }
    blouse_columns = np.array( [ 'neckline_left', 'neckline_right',\
       'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left',\
       'armpit_right',  'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left',\
       'top_hem_right'] )

    outwear_columns =  np.array( ['neckline_left', 'neckline_right',\
         'shoulder_left', 'shoulder_right', 'armpit_left',\
           'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',\
           'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left',\
           'top_hem_right'] )

    trousers_columns = np.array( [ 'waistband_left', 'waistband_right',  'crotch', 'bottom_left_in', 'bottom_left_out',\
           'bottom_right_in', 'bottom_right_out'] )

    skirt_columns = np.array( [ 'waistband_left', 'waistband_right',  'hemline_left', 'hemline_right'] )

    dress_columns = np.array( ['neckline_left', 'neckline_right',\
           'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left',\
           'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in',\
           'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'hemline_left',\
           'hemline_right'] )

    cates = np.array([blouse_columns,outwear_columns,trousers_columns,skirt_columns,dress_columns])

    assert (type(i) == str or type(i) ==int) , "wrong arguement type "

    if type(i) == int:
        return cates[i]
    else:
        return cates[cates_map[i]]

def get_cate_name(i):
    assert (type(i) == str or type(i) ==int) , "wrong arguement type "
    cates_map = ["blouse" ,"outwear","trousers","skirt","dress" ]
    if type(i) == int:
        return cates_map[i]
    else:
        assert (i in cates_map), "wrong spell"
        return i