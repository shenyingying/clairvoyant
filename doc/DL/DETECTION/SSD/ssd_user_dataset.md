#labelImg annotation pic
 1.tool:[labelImg](https://github.com/tzutalin/labelImg)
   
     tips：学习下安装，快捷键可以达到事半功倍的效果。
   
 2.我的标注过程：
   ![label](/home/sy/paper/write/pic/markdown/python/label.png)
   我的标注结果：
   ![label_result](/home/sy/paper/write/pic/markdown/python/label_result.png)
# {}.txt convert {}.xml
  `python txt_xml()` [code](/home/sy/code/script/clairvoyant/trans.py)
# {}.xml convert {}.cvs
   `python xml_csv()` [code](/home/sy/code/script/clairvoyant/xml_to_csv.py)
   ##csv_result
   ![csv_result](/home/sy/paper/write/pic/markdown/python/csv_result.png)
# {}.csv convert {}.record
  ``os.chdir('/home/sy/code/project/models/research/')``
   `--csv_input=data/tv_vehicle_labels.csv  --output_path=train.record ` [code](/home/sy/code/script/clairvoyant/generate_tfrecord.py)
   
    tips:因为无法马上验证结果，一定要用python3转换{我花了两天时间才得出这个结论}