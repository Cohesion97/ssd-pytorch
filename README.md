# ssd-pytorch
目前还是个半成品
使用run.sh进行训练（DDP）
使用test.py生成测试结果（pkl文件）
使用eval.py计算mAP

## 当前性能
|训练集|测试集|训练集loss|mAP|备注|
|:---:|:---:|:---:|:---:|:---:|
|VOC2007trainval|VOC2007test|4.05|45.39|with maxioucut without warmup|
|VOC2007trainval|VOC2007test|0.23|49.95|without maxioucut with warmup|
|VOC2007trainval|VOC2007test|1.5|38.85|without maxioucut without warmup|

## 包含功能
* data-augmentation
* warm-up

## TODO
+ [ ] 模型参数封装
* [ ] 训练过程中加入测试

