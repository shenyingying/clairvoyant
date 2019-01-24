
获得	1
移植 need do：2018/12/11	1
1.tensorflow lite 移植	1
Tensorflow question:	6
1. Failed to get convolution algorithm. This is probably because cuDNN failed to initialize	6
2.An unexpected error has occurred. Conda has prepared the above report.Solving environment: failed	6
NVIDA GTX2080 Ubuntu18.04 cuda10.0 cudnn7.5 tensorflow 1.12 install	7
Andriod studio 导入项目中常见问题：	11
Q1：You can move the version from the manifest to the defaultConfig in the build	11
Q2：Could not find com.android.tools.build:aapt2:3.2.1-4818971.	12
Q3：错误: 找不到符号private final Fill<T> fill;	12
LIUNX INSTALL caffe	13
Need ask android engineer	14
Snpe progress	14
获得
1. darknet .c 在手机端 CPU 可以获得 1.2s
2. SNPE 官网程序没有跑通  
3. Tile mobilenet 70ms

移植 need do：2018/12/11
1.tensorflow lite 移植
A：把模型.weight保存成.pb文件：
    1 .weight->.h5 convert.py
$ python convert.py pupils_tiny.cfg pupils_tiny_final.weights model_data /pupils_tiny.h5


    2 .h5->.pb h5_pb.py
https://github.com/amir-abdi/keras_to_tensorflow
--input_model /home/bluesandals/sourceCode/keras-yolo3/model_data/pupils_tiny.h5 --output_model /home/bluesandals/sourceCode/keras-yolo3/model_data/pupils_tiny.pb

3  .pb android studio   
总结：因为.pb文件输出的都是层，无法写java主程序；寻找检测的title的案例。
A1；I wonder if there is something wrong in trans 
https://github.com/mystic123/tensorflow-yolo-v3

A1：把模型.ckpt保存成.pb文件：
Checkpoint 检查点文件，文件保存了一个目录下所有的模型文件列表
Model.ckpt.meta文件保存了Tensorflow计算图的结构，可以理解为神经网络的网络结构，该文件可以被tf.train.import_meta_graph加载到当前默认的图来使用
Ckpt.data保存模型中每个变量的取值
Tf.traon.write_graph():默认情况下只导出网络定义（没有权重）
Tf.train.Saver().save():到处文件graph_def与权重分离，
                      graph_def中没有Variable只有constant
                  若把权重variable转换成constant 可实现一个文件中包含权重网络
Convert_variables_to_constants():固化模型，将计算图中的变量取值以常量形式保存。
转换思路：
通过传入CKPT模型的路径得到模型的图和变量数据
通过import_meta_graph导入模型中的图
通过saver.restore从模型中恢复图中各个变量
通过Convert_variables_to_constants()将模型持久化



B. tensorflow android老版本移植
/tensorflow-master\tensorflow\examples\android
遇到的问题：
Q1：T扩展已经在类Zeros中生声明的object

A1：

Q2:you shouldn’t define default version in Manifest.xml,you should define it in build.gradle
A2:

Done :there is three app in you phone,how to use you model,it’s seems like some trouble

B1:新版本中没有yolo mode，so need to back old version:
记录detecter mode= YOLO :
Case 1: voc label->tiny-yolo-voc.pb: work;
Case2: coco label->yolov2-tiny.pb :(change point)
One: the file directory (your self convert)
Two:the string label
Three: the num classes
Case 3:coco label -> yolov3-tiny.pb 
One: print the yolov2-tiny.pb and yolov3-tiny.pb 
Two:check the source code,read the paper,change the code
B2：darkflow 转换命令
flow --model yolov3.cfg --load yolov3.weights --savepb 
Q1：如何把YOLOV3.weight->yolov3.pb(AS demo can recognition)
     Try to code a little demo？
     高通提供转换模块？

1. 修改darkflow源码，适应yolov3
2. 用qqww Kerase 重新训练 直接保存成 修改源码直接保存成.pb文件
3. Tensorflow yolov3

Sure the version confliction

// Copy the input data into TensorFlow.


2. Ncnn yolo3 移植
   Mld 是百度的不敢用，title的原主谷歌在搞分布式计算，ncnn腾讯社区不错
C:Ubuntu 下gcc 出现问题
Q1：E: Sub-process /usr/bin/dpkg returned an error code (1)
https://cn.aliyun.com/jiaocheng/1384266.html
Cmd：
sudo mv /var/lib/dpkg/info /var/lib/dpkg/info.bak
sudo mkdir /var/lib/dpkg/info
Sudo apt-get update
Q2：cd /usr/bin/ 删除所有的gcc g++
重新安装 gcc g++  
https://blog.csdn.net/csdn_zhishui/article/details/83751120
Cd/usr/bin
Sudo rm gcc*
Sudo rm g++*
Sudo apt install gcc-5
Sudo apt install g++-5
Sudo ln -s gcc-5 gcc
Sudo ln -s g++-5 g++




Tensorflow question:
1. Failed to get convolution algorithm. This is probably because cuDNN failed to initialize

 Question:
 A: 确认cuda 安装好 （为什么gcc 降级了，终端的gcc 没有降级） 
 B：把终端的python 改成python3.6 
 C：终极绝招--安装anaconda 
D:tensorflow版本太高 安装低版本
E:cudnn 版本不对应 安装对应版本，我升级了cudnn7.4

2.An unexpected error has occurred. Conda has prepared the above report.Solving environment: failed

关闭翻墙从新尝试 it’s works 

3.CondaHTTPError: HTTP 404 NOT FOUND for url <https://pypi.tuna.tsinghua.edu.cn/simple/noarch/repodata.json>
Elapsed: 00:00.345942(为什么会出现这个连接，刚才我在pycharm 中更改源目录的时候添加上去的)

Answer: 找到。.condarc文件所在位置直接删除找不到的连接

3. 现在最新的anconda 默认python是3.7 在 base 下直接 conda install tensorflow -gpu 一直处于搜索状态 (没有3.7 支持的tensorflow)
4. https://pypi.python.org/simple


NVIDA GTX2080 Ubuntu18.04 cuda10.0 cudnn7.5 tensorflow 1.12 install
心仪已久的显卡2080终于到了，怎么Ubuntu18那么帅，怎么还能容忍16，14横行
废话不说，直接捞干的来
1. Ubuntu18 安装 不用说，不过有blog 写到 如果在Ubuntu18下安装NVIDIA 的驱动 在install Ubuntu 不要直接enter 学要按着e 修改配置 just like this    
https://blog.csdn.net/tjuyanming/article/details/79267984 ,我尝试了没有成功，就放弃了，直接在软件更新里安装：

还是很感谢这个博主 https://blog.csdn.net/tjuyanming/article/details/80862290
不知道为什么我的机器检测显卡 没有出现显卡型号（有哪位大神知道原因告知下）：

2. 安装cuda，因为Ubuntu18 只能下载到 cuda10.0，按着正常套路忐忑着继续往下走：

3. 安装cudnn7.5：

4. 安装tensorflow（中间走了好多弯路）：
  A：命令行直接安装  pip3 install tensorflow-gpu
验证测试的时候会出现：

经过一番查找，得知默认安装最近版本的tensorflow(1.12版本2018.11.29)最高支持到cuda9.2，所以与安装的cuda10.0相矛盾，怎么办，好多办法，我的原则是不放弃Ubuntu18就不放弃 cuda10.0 额 额 ，这时候anaconda3 华丽的出场了
5. 安装anaconda请参考这个下面
                                      https://blog.csdn.net/qq_31119155/article/details/80793006
6. 在conda中建立环境 安装tensorflow：
  A: conda create -n sy(your file name) python=3.6
  B:conda activate sy
  C:conda install tensorflow-gpu
  (其实conda环境中重现安装了cuda9*，所以可以运行成功)

  D: 验证测试：

Ps:Ubuntu18 默认是python3.6 如果你在运行比较老的代码的时候 需要python2.7 你完全可以在conda 下新建立个环境 制定python （因为我发现安装上anacoda之后 再用超联结的方法更换python2.7 和 pythpn 3 已经不好用了），最近在玩SNPE，有没有一块的伙伴，头大ing
7. 环境配置好，如何进行导入IDE进行源码编写 ，可以参考
https://blog.csdn.net/qq_31119155/article/details/83992500
鼓捣了cuda10，好像这个方法不太妙，没关系，我们在cuda下不是有好多环境的吗？选择一个不是cuda10的就ok（因为目前tensorflow还不支持cuda10，期待tensorflow跟上nvidia的步伐啊），具体操作
A：先确定你哪个环境可以用

B:导入编译器


Result:


Andriod studio 导入项目中常见问题：

Q1：You can move the version from the manifest to the defaultConfig in the build

Answer：

Q2：Could not find com.android.tools.build:aapt2:3.2.1-4818971.

Answer:

Q3：错误: 找不到符号private final Fill<T> fill;

https://github.com/tensorflow/tensorflow/issues/21431
// set to 'bazel', 'cmake', 'makefile', 'none'
def nativeBuildSystem = 'none'
LIUNX INSTALL caffe-GPU
1. 修改MakeFile.config 文件
A:将 #USE_CUDNN := 1 修改成： USE_CUDNN := 1
B:将 #OPENCV_VERSION := 3 修改为： OPENCV_VERSION := 3
C:将 #WITH_PYTHON_LAYER := 1 修改为 WITH_PYTHON_LAYER := 1
D:INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib 
修改为： 
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu/hdf5/serial       
   
2. 修改MakeFile文件
A:
将：
NVCCFLAGS +=-ccbin=$(CXX) -Xcompiler-fPIC $(COMMON_FLAGS)
替换为：
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)
B:
将：
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_hl hdf5
改为：
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
3. 修改 /usr/local/cuda/include/host_config.h
若找不到进入host_config.h中include的文件
将
#error-- unsupported GNU version! gcc versions later than 6.0 are not supported!
改为
//#error-- unsupported GNU version! gcc versions later than 6.0 are not supported!

4.make all -j8
4. 修复bug：
nvcc fatal : Unsupported gpu architecture 'compute_20'
解决办法：注释掉MakeFile.config 文件中的前两句
CUDA_ARCH := #-gencode arch=compute_20,code=sm_20 \ 
#-gencode arch=compute_20,code=sm_21 \ 
-gencode arch=compute_30,code=sm_30 \ 
-gencode arch=compute_35,code=sm_35 \ 
-gencode arch=compute_50,code=sm_50 \ 
-gencode arch=compute_52,code=sm_52 \ 
-gencode arch=compute_60,code=sm_60 \ 
-gencode arch=compute_61,code=sm_61 \ 
-gencode arch=compute_61,code=compute_61
不支持compute_20 即把compute_20 所在行注释掉，然后make clean，重新make

参考文档：https://blog.csdn.net/yhaolpz/article/details/71375762
          https://github.com/BVLC/caffe
LIUNX INSTALL caffe-CPU
1. 安装依赖
sudo apt-get install libprotobuf-dev 
sudo apt-get install libleveldb-dev
sudo apt-get install libsnappy-dev 
sudo apt-get install libopencv-dev
sudo apt-get install libhdf5-serial-dev
sudo apt-get install protobuf-compiler
sudo apt-get install libgflags-dev
sudo apt-get install libgoogle-glog-dev
sudo apt-get install liblmdb-dev
sudo apt-get install libatlas-base-dev
2. 下载caffe
sudo apt-get install git
git clone git://github.com/BVLC/caffe.git
3. 编译caffe

A:
cd caffe/
cp Makefile.config.example Makefile.config
sudo gedit Makefile.config
B:
去掉CPU_ONLY前面的#号

使得CPU_ONLY := 1
C:
配置引用文件路径（主要是HDF5的路径问题）

修改为：

新增内容为：
/usr/include/hdf5/serial 
/usr/lib/x86_64-linux-gnu/hdf5/serial
D：
执行编译命令
sudo make all
sudo make test 
sudo make runtest
E:运行终端配置文件导入终端
Sudo apt-get install gfortran
Cd caffe/python
 for req in $(cat requirements.txt);do pip install &req;done
Sudo pip install -r requiremnts.txt
Sudo gedit ~/.bashrc
Export PYTHONPATH=/home/bluesandals/caffe/python:$PYTHONPATH
Sudo ldconfig
Cd caffe/
Make pycaffe

F：可能错误
Error:
①　 .build_release/lib/libcaffe.so：对‘cv::imdecode(cv::_InputArray const&, int)’未定义的引用
Resolve:
查看opencv配置
pkg-config --modversion opencv 
修改MakeFile.config文件中：解注释 use-opencv 
②　 Sudo pip install -r requiremnts.txt

需要重新安装 leveldb, python-dateutil
Leveldb 安装：https://github.com/google/leveldb
python-dateutil 安装： pip install python-dateutil
LIUNX INSTALL pytorch-CPU
1. 去官网https://pytorch.org/get-started/locally/找命令执行
2. 终端执行命令验证是否安装成功
  Python
  Import torch
  Import torchvision







1


Need ask android engineer（first）
1. the qualcomm bug (show the pic)
[显示的问题，调节字体大小]
2. How to BreakPoint debug / the title office demo (yolov3)
3. The AS structure etc (recording duration)
（Second）
1. 验证yolov3_tiny.weight->yolov3_tiny.pb 转换是否成功->lite
2. 验证.dlc转换是否成功，MobileNet_SSD->修改源码imageClassify输出
3. 验证yolov3.pb->yolov3.dlc转换是否成功
4. 解析android的大致架构；


Snpe progress
1. Select the neural network model and runtime target
2. Create one or more input tensor(s)
3. Populate one or more input tensor(s) with the network input(s)
4. Forward propagate the input tensor(s) through the network
5. Process the network output tensor(s)

SNPE API 使用：
1. snpe-tensorflow-to-dlc
 --graph models/inception_v3/tensorflow/inception_v3_2016_08_28_frozen_opt.pb 
--input_dim  input 1,299,299,3 
--out_node InceptionV3/Predictions/Reshape_1 
--dlc inception.dlc --allow_unconsumed_nodes
file:///home/bluesandals/software/snpe-1.21.0/doc/html/tools.html#tools_snpe-tensorflow-to-dlc
snpe-tensorflow-to-dlc
--graph models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb
-i Preprocessor/sub 1,300,300,3
--out_node detection_classes
--out_node detection_boxes 
--out_node detection_scores 
--dlc mobilenet_ssd.dlc 
--allow_unconsumed_nodes
snpe-caffe-to-dlc
--caffe_txt models/MobileNet_deploy.prototxt 
--caffe_bin models/MobileNet_deploy.caffemodel 
--dlc caffe_model_ssd.dlc
Yolov2-tiny.weitht+yolov2-tiny.cfg->yolov2-tiny.pb
    (darkflow+read-pd(print_pb_name.py))
Yolov2-tiny.pb->yolov2-tiny.dlc
    （不知是否成功？目前无法验证）
Yolov3.weight->yolov3-tiny.pb
   （用tensorflow lite验证 tensorflow-yolo-v3中的转换是否对的 ）
     https://github.com/mystic123/tensorflow-yolo-v3

    






