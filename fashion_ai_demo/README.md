# FashionAI全球挑战赛服饰关键点定位
### 队名： 学习一个
### 排名： 78名


**更新笔记**

这次提交和上次email的提交更新了以下几点：
1. 改变了几个代码文件目录
2. 在训练和测试脚本中补充了更详细的命令行参数
3. 在训练和测试脚本中修改了读图片的目录
4. 加入了validation集的图片
5. 将README改的更详细

系统：
ubuntu 16.04 64bit

环境:  
python 3.6.3  
CUDA 9.0  
CuDNN  

库:  
numpy (1.14.0)  
pandas (0.20.3)  
Pillow (5.0.0)  
scikit-image (0.13.0)  
scipy (0.19.1)  
tensorflow-gpu (1.5.0)  


```
目标：
|--util 功能文件夹
    |-- preprocess.py 图片预处理的函数
    |-- categories.py 返回不同类型的关键点名字

|--train_code
    |--preprocess_train.py 预处理训练集图片的脚本
    |--cpm_train.py 训练模型脚本

|--test_code
    |--preprocess_test.py 预处理测试集图片的脚本
    |--cpm_pred.py 通过读取checkpoint参数预测图片关键点   
    |--data_to_submit_format.py 将各类的结果汇总成为提交的格式

|--model
    |--cpm.py 模型架构代码
    |--train_input.py 读取图片和坐标的代码，并将图片转换为数组，坐标变为热度图

|--train 训练集图片和Annotation文件

|--test 测试集图片和图片表格文件

|--valid 验证集图片和Annotation文件
```


核心算法: 

Convolutional Pose Machines[1]
Reference:[1] Wei, S. (2016). Convolutional Pose Machines. arXiv:1602.00134 [Cs]. Retrieved from http://arxiv.org/abs/1602.00134

训练集: train_set的所有图片和warm_up中不为验证集中的图片
验证集： 从warm up图片各个类型选取 200张图片
测试集： test set中的图

训练时的超参数设定：
* 超参数设定为:
* 总迭代数:40000 （每5000步存一次checkpoint）
* batch size :5 
* learning rate 0.0001 (每5000步decay 0.5)
* drop out保留率： 0.5 (只用在最后一层)
* L2正则 lambda: 0.001

模型
输入： 训练集的RGB图片 维度：[训练集大小或batch大小,512,512,3]
输出:  各个关键点的热度图 维度： [训练集大小或batch大小,64,64,关键点个数]。
损失函数： 预测热度图 和 ground truth热度图像素之间root-mean-square error

Pipeline:
1. 预处理数据

    * 目的： 将尺寸不为512\*512的图片 pad为 512\*512，并更新padding后关键点的坐标值。
    * 脚本流程:

        执行./train/preprocess_train.py  和 ./test/preprocess_test.py 后会生成 ./train_pad, ./valid_pad 和 ./test_pad文件夹 并存入pad后的图片为 512\*512，以及按照类型分类的图片和坐标表格。


2. 训练模型

    * `python train_code/cpm_train.py [类型] [图大小] [训练集大小] [-l 学习率] [-k L2 lambda] [-k drop out 保留率]`

        （训练集大小为-1时，为使用所有的训练集，--help来看帮助文件）

    * 通过train_input.py 中的`data_stream`类读取训练集和验证集图片的文件名和关键点坐标。
每一次迭代，data_stream对象随机选择batch （这里是5）个 图片， 返回图片作为input，关键点的热度图作为label，导入神经网络进行训练。并在一定步数的时候存checkpoint和Log
        (Demo中总迭代数:50 并且每50步存一次checkpoint)

    * 训练的参数和日志存在： 
        * ./params/CPM/[类型]/l2_[L2 lambda]_drop_[保留率]/

        * ./log/train_log/CPM/[类型]/l2_[L2 lambda]_drop_[保留率]/


3. 预测模型

    * 运行python test_code/cpm_pred [类型] [图大小] [测试集大小]。
    * 通过train_input.py 中的data_stream类读取测试集图片的文件名。然后读取相对应的checkpoint文件，checkpoint文件名为`config` 类中的`self.load_filename`。
预测出各个关键点热度图，并取图中最大值为该关键点坐标。
输出图片名，类型以及预测出的坐标到/result/cpm_[类型]_图片大小.csv
最后调用data_to_submit_format.py 将各类结果汇总为提交的格式








