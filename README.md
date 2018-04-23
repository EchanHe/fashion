# fashion
环境:
python 3.6.3
CUDA
CuDNN

库:
numpy (1.14.0)
pandas (0.20.3)
Pillow (5.0.0)
scikit-image (0.13.0)
scipy (0.19.1)
tensorflow-tensorboard (1.5.1)

Submit_code
|--util 功能文件夹
    |-- preprocess.py 图片预处理的函数
    |-- categories.py 返回不同类型的关键点名字

|--train_code
    |--preprocess_train.py 预处理训练集图片的脚本
    |--cpm_train.py 训练模型脚本
    |--cpm.py 模型架构代码
    |--train_input.py 读取图片和坐标信息的代码

|--test_code
    |--preprocess_test.py 预处理测试集图片的脚本
    |--cpm_pred.py 通过读取checkpoint参数预测图片关键点
    |--cpm.py 模型架构代码
    |--train_input.py 读取图片和坐标信息的代码

方法: Convolutional Pose Machines[1]
Reference:[1] Wei, S. (2016). Convolutional Pose Machines. arXiv:1602.00134 [Cs]. Retrieved from http://arxiv.org/abs/1602.00134

训练集: train_set的所有图片和warm_up中不为验证集中的图片
验证集： 从warm up图片各个类型选取 200张图片
测试集： test set中的图

Pipeline:
预处理数据
假设train 和 test 图片加压至根目录./train ./test.
执行./preprocess_train.py  和 ./preprocess_test.py 后
会生成 ./train_pad 和 ./test_pad文件夹 并存入pad后的图片为 512\*512，以及按照类型分类的图片和坐标表格。

训练模型

输入： 训练集的RGB图片 维度：[训练集大小或batch大小,512,512,3]
输出:  各个关键点的热度图 维度： [训练集大小或batch大小,64,64,关键点个数]

读取train 和 validation 的数据
通过控制台设置训练脚本训练的类型。
python cpm_train [类型] [图大小] [训练集大小] [-l 学习率] [-k L2 lambda] [-k drop out 保留率]
训练的参数存在： 
~/params/CPM/[类型]/l2_[L2 lambda]_drop_[保留率]/

训练日志：
~/log/train_log/CPM/[类型]/l2_[L2 lambda]_drop_[保留率]/


预测模型。
读取test的数据
读取相对应的checkpoint文件。
预测出各个关键点热度图，并取图中最大值为该关键点坐标。输出结果文件至：~/result/cpm_[类型]\_[图片大小].csv








