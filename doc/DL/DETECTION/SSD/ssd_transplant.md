# convert .pb to .dlc
  
    python3 object_detection/export_inference_graph.py --input_type=image_tensor --pipeline_config_path=/home/sy/code/project/models/research/object_detection/training/pipeline.config --trained_checkpoint_prefix=/home/sy/code/project/models/research/object_detection/training/model.ckpt-15432 --output_directory=/home/sy/data/work/StandardCVSXImages/log_old/

   tips: 因为snpe office 只支持到tf-1.6,而tf office 只有 tf-1.9之上的，需要用老版本训练，转化不能成功转化。
   
# some change in office demo code
  ## 屏蔽权限
   1. permission question
  
   ![permission](pic/permission.png)
   ans: 移走 ～/src/main/res/raw/inception_v3.zip，把改文件移动到chip中
         
    mv /home/sy/code/project/snpe-1.22.0.212/examples/android/image-classifiers/app/src/main/res/raw/inception_v3.zip <path>/inception_v3.zip
    tar -zxvf <path>/inception_v3.zip
    adb shell
    cd /storage/emulated/0/Android/data/com.qualcomm.qti.snpe.imageclassifiers/files/models
    mkdir inception_v3
    adb push <path>/inception_v3 /storage/emulated/0/Android/data/com.qualcomm.qti.snpe.imageclassifiers/files/models/inception_v3
      
   adb push:
   
   ![adbpush](pic/adb_push.png)
     
   chip content:
   
   ![chip](pic/chip.png)
    
   在com.qualcomm.qti.snpe.imageclassifiers.tasks.LoadModelsTask.java 64行 加
     `availableModels.add("inception_v3");` 
  
  ## 修改显示
  这个问题困扰好久（近一个月），还天真的以为官网的demo有bug，怎么没有显示，无知真可怕，最后app工程师出面解决，知晓一种语言是多么重要。
  其实很简单的修改显示设置：
  
    layout/fragment_model.xml 中长宽显示
  ![show](pic/show.png)
  
  ## mobilenet_ssd specials
  
  
  

# result in 820 chip

# accuracy compare between pc and chip