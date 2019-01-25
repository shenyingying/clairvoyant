YOLO 训练自己数据
花了近一个星期的时间捣鼓，用yolo训练自己的数据，参考了几十篇博客，发现好多坑啊，尤其是CSDN上的博客，说多了都是泪啊，闲话少扯，我们直接进入正题。
1.根据yolo官网指令跑起来（没毛病） https://pjreddie.com/darknet/yolo/
2.修改makefile 文件：
   A：  GPU=1 （GPU faster CPU 500倍）
CUDNN=1
OPENCV=1
   B：ARCH= -gencode arch=compute_30,code=sm_30 \
      -gencode arch=compute_32,code=sm_32 \
      -gencode arch=compute_30,code=[sm_30,compute_30] \
      -gencode arch=compute_32,code=[sm_32,compute_32]  （你的N卡）
   C：NVCC=/usr/local/cuda-9.0/bin/nvcc （你的版本）
   D： ifeq ($(GPU), 1) 
COMMON+= -DGPU -I/usr/local/cuda-9.0/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda-9.0/lib64 -lcuda -lcudart -lcublas -lcurand
Endif （适配你电脑的型号）
3.标注文件 labelImg 特别好用 直接出 yolo所需要的.txt文件，我标注了两类
0 0.392708 0.499219 0.110417 0.026562
0 0.636458 0.520312 0.122917 0.021875
1 0.392708 0.496875 0.014583 0.009375
1 0.639583 0.518750 0.012500 0.006250
一行五个数字分别为类母，中心坐标 x y 长宽 w h 
在你喜欢的地方保存这些图片和标注文件，并建立一个train.txt 和val.txt，这个文件的地址4要用到，并把图片的地址写入其中。
这里我深深的被坑了，建立文件夹下面（image，label）文件，image文件夹下面保存图片和.txt ; label文件夹下面保存.txt.
4.修改配置文件，copy cfg/yolov3.cfg 为 face.cfg 并做如下修改：
   修改的地方:
   A: 若测试：
# batch=1
# subdivisions=1
若训练：
# batch=64
# subdivisions=8 （根据你N卡计算能力来）
   B：修改YOLO上面最近一个 [convolutional]中的filter=[class+5]*3 (共有三处)
   C：修改YOLO层中classes (共有三处)
5.在data下新建立 face.names 文件 写入你要标记的类别 just like
    eye
Pupils
6.在cfg 下新建 face.date 写入：
classes= 2
train  = /home/bluesandals/code/darknet/face_data/train.txt （地址随意）
valid  = /home/bluesandals/code/darknet/face_data/val.txt （地址随意）
names = data/face.names （上面新建的.names文件）
backup = backup （输出权重文件地址 也可以为 result）
7.对.cfg文件中各个参数说明：
  [net]
# Testing
# batch=1
# subdivisions=1
# Training
batch=32 # 一批训练样本的样本数量，每batch个样本更新一次参数
subdivisions=16 #batch/subdivision作为一次性送入训练器的样本数量
# 每轮迭代会从所有训练集中随机抽取batch=32个样本参与训练，所有这些batch的样本又会分为 # ubdivision次送入网络，已减轻内存占用的压力）　
width=608
height=608
channels=3
momentum=0.9  # 动量 前后梯度一致加速学习，前后不一致，抑制震荡
decay=0.0005  # 正则项，避免过拟合
angle=0   # 数据扩充时，图片旋转的角度
saturation = 1.5 # 饱和度范围
exposure = 1.5  # 曝光度范围
hue=.1  # 色调变化范围

learning_rate=0.001 # 初始学习率，开始（0.1～0.001），后来应指数衰减（100倍）
burn_in=1000  # 迭代次数小于burn_in时，学习率有一种更新方式，当大于采用policy
max_batches = 500200 # 训练达到 max_batches 训练停止
policy=steps 
steps=400000,450000
scales=.1,.1

[convolutional]
batch_normalize=1  #是否做BN
filters=32         #输出特征图的数量
size=3             #卷积核的尺寸
stride=1           #做卷积运算的步长
pad=1  # 如果pad=0,padding 由padding参数制定，如果pad=1,padding=size/2
activation=leaky #激活函数

# Downsample

[convolutional]
batch_normalize=1
filters=64
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=32
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3  #shortcut 部分是卷积的跨层连接，就像Resnet中使用的一样，from=-3,意思是shortcut的输出是通过与先前倒数第三层相加再输入之后的网络
activation=linear
输入与输出：保持一致，并且不进行其他操作，只是求差。
处理操作：res层来源于resnet，为类解决网络的梯度弥散或者梯度爆炸的现象，提出将深层次的神经网络的逐层训练该为逐阶段训练，将深层神经网络分为若干个子段，每个小段包含比较浅的网络层数，然后用shortcut连接方式使得每个小段对于残差进行训练，每一个小段学习总差（总的损失）的一部分，最终达到总体较小
# Downsample

[convolutional]
batch_normalize=1
filters=128
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=64
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=256
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=512
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear


[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

# Downsample

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=2
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
filters=1024
size=3
stride=1
pad=1
activation=leaky

[shortcut]
from=-3
activation=linear

######################

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky

[convolutional]
batch_normalize=1
filters=512
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=1024
activation=leaky
stopbackward=1 # 提升训练速度

[convolutional]
size=1
stride=1
pad=1
filters=21  # 每一个yolo层前最后一个卷积层 filters=(classes+1+coords)*anchors_num
activation=linear


[yolo]
mask = 6,7,8
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
# anchors是可以通过事先cmd指令计算出来的，和图片数量，width,height,cluster有关，也可以# 手工挑选，也可以通过kmeans从训练样本集中学出。
classes=2
num=9 #每个grid cell 预测的box，和anchor的数量一致，当要使用更多的anchors需要增大num，##若增大num后训练时，obj～0,可以尝试调大object_scale
jitter=.3 # 利用数据抖动产生更多的数据
ignore_thresh = .7 #决定是否需要计算IOU误差参数，大于thresh,IOU误差参数不计算在cost中
truth_thresh = 1
random=0 # 如果为1,每次迭代图片大小随机从320～608,部长为32,如果为0 每次训练输入大小一致


[route]
layers = -4

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 61



[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
batch_normalize=1
filters=256
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=512
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear


[yolo]
mask = 3,4,5
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=2
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=0



[route]
layers = -4

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[upsample]
stride=2

[route]
layers = -1, 36



[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
batch_normalize=1
filters=128
size=1
stride=1
pad=1
activation=leaky

[convolutional]
batch_normalize=1
size=3
stride=1
pad=1
filters=256
activation=leaky

[convolutional]
size=1
stride=1
pad=1
filters=21
activation=linear


[yolo]
mask = 0,1,2
anchors = 10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326
classes=2
num=9
jitter=.3
ignore_thresh = .7
truth_thresh = 1
random=0

# 批输出 1491: 2.497199, 2.531538 avg, 0.001000 rate, 12.053180 seconds, 47712 images
#  1491：当前训练的迭代次数
#  2.497199：总体loss
#  2.531538 avg:平均loss（这个值应该是越低越好，一般来说，一旦这个数低于0.060730 训练可以终止）
#  0.001000 rate：当前学习率 在.cfg文件中定义了初始值和学习策略
#  12.053180 seconds：该批次训练所花费的总时间
#  47712 images：到目前为止，参与训练图片的总量=迭代次数（1491）×batch（32）
#  Region 106 Avg IOU: 0.498723, Class: 0.998587, Obj: 0.022303, No Obj: 0.000196, .5R: 0.500000, .75R: 0.250000,  count: 8;
# 三个尺度上预测不同大小的框;
# Region 82 Avg IOU:卷积层为最大的预测尺度，使用较大的mask，预测出较小的物体;
# Region 94 Avg IOU:卷积层为中等的预测尺度，使用中等的mask，预测出中等的物体;
# Region 106 Avg IOU:卷积层为最小的预测尺度，使用较小的mask，预测出较大的物体;
#  Avg IOU: 0.498723:代表当前subdivision内的图片的平均IOU，代表预测的矩形框和真实目标框的交集与并集之比，0.9已经很高率;
#  Class: 0.998587：标注物体分类的正确率，期望该值趋近于1;
#  Obj: 0.022303：越接近1越好;
#  No Obj: 0.000196：期望该值悦来越小，但不为0;
#  .5R: 0.500000：以IOU=0.5为阈值的时候的recall;recall=检出的正样本/实际正样本
#  .75R: 0.250000：以IOU=0.75为阈值的时候recall;
#  count: 8:所有的当前subdivision图片中包含正样本图片的数量。
训练的时候需要输出日志文件：
./darknet detector train cfg/face.data cfg/face.cfg darknet53.conv.74 -gpus 0,1 2>1 | tee face_train.log
测试训练结果：
./darknet detector test cfg/face.data cfg/face.cfg face_900.weight face_data/0000001.jpg -thresh 0.25
验证集：标记好的数据，在训练的过程中不参与训练，验证算法通过对比预测目标和标记目标判断预测的正确率，评价模型;
acc：正确数量/（正确数量+错误数量）
./darknet detector recall cfg/face.data cfg/face.cfg backup/face_final.weight
./darknet detector valid cfg/face.data cfg/face.cfg backup/face_final.weight -out 123 -gpu 0 -thresh .5