| model | train strategy  | resolution | loss |  map  |detection execute time  |cvs execute time|
|:----:|:----:|:----:|:----:|:------:|:------:|:------:|
|  ssd   |  from script   |    224*224   |  **    |  **      |  80ms        |   410ms     |
|  yolo  | from yolo_weight.h5     |   416*416  |   11.7795    |    100%      |  840ms | 4.9s   |
|  yolo  | from yolo_416_weight.h5 |  128*128   |   10.7129    |    98.38%    |  200ms | 950ms  |
|  yolo  | from yolo_416_weight.h5 |  96*96     |   11.1494    |    97.37%    |  189ms | 840ms  |
|  yolo_mobilenetV1_alpha=0.75   |  from imagenet | 320*320 |  6.3284     |    100%    | 300ms  | 1.6s   |
|  yolo_mobilenetv2_alpha=0.75   |  yolo_mobilenet| 320*320 |  4.2952    |    100%     | 280ms  | 1.1s   |
|  yolo_mobilenetv2_alpha=0.75   |  yolo_mobilenet| 320*256 |  3.4265    |    100%     | 未解析成功 |     |
|  yolo_mobilenetv2_alpha=0.5_out1  |  yolo_mobilenet| 224*224 |  6.5603 |    100%     |  100ms    |  750ms   |
|  yolo_mobilenetv2_alpha=0.5_out1  |  yolo_mobilenet| 192*192 |  3.9101 |    100%     |   110ms    |  660ms   |
|  yolo_mobilenetv2_alpha=0.75_out1 |  yolo_mobilenet_shortlayer| 224*224 |  3.8901    |    100%    |   90ms  |  600ms   |
|  yolo_mobilenetv2_alpha=0.5_out1 |  yolo_mobilenet_shortlayer| 224*224 |  3.8901    |    100%    |   75ms  |     |
|  yolo_efficientNets   | from imagenet     |320*320  |  9.1328      |    98.23%  |  未转化成功 |