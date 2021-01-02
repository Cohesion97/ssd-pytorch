# ssd-pytorch
目前还是个半成品
* 使用run.sh进行训练（DDP）
* 使用test.py生成测试结果（pkl文件）
    * nms使用mmcv提供的代码
* 使用eval.py计算mAP

## 当前性能
|训练集|测试集|训练集loss|mAP|备注|
|:---:|:---:|:---:|:---:|:---:|
|VOC2007train|VOC2007test|4.05|45.39|with minioucut without warmup|
|VOC2007train|VOC2007test|0.23|49.95|without minioucut with warmup|
|VOC2007train|VOC2007test|1.5|38.85|without minioucut without warmup|
|VOC2007train|VOC2007test|2.7|63.79|with minioucut with warmup|

## 包含功能
* data-augmentation
* warm-up

## TODO
+ [ ] 模型参数封装
* [ ] 训练过程中加入测试

