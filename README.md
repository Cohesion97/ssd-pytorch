# ssd-pytorch
目前还是个半成品
* 使用run.sh进行训练（DDP）
* 使用test.py生成测试结果（pkl文件）
    * nms使用mmcv提供的代码
* 使用eval.py计算mAP

## 当前性能
|训练集|训练集loss|mAP|minIoUCut|200 step warmup|
|:---:|:---:|:---:|:---:|:---:|
|VOC2007train|4.05|45.39| <input type="checkbox"> |<input type="checkbox"> |
|VOC2007train|0.23|49.95|<input type="checkbox"> |<input type="checkbox"> |
|VOC2007train|1.5|38.85|<input type="checkbox"> |<input type="checkbox"> |
|VOC2007train|2.7|63.79|<input type="checkbox">|<input type="checkbox"> |

## 包含功能
* data-augmentation
* warm-up

## TODO
+ [ ] 模型参数封装
* [ ] 训练过程中加入测试

