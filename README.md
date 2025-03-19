# agri-dino
项目代码见master，数据使用的是工业大麻雄蕊图片，标注文件已放在stamen_dataset文件夹，在stamen_dataset内新建**images**文件夹，内部放置自己的数据集（train、val、test），下载*groundingdino官网*的初始权重groundingdino_swint_ogc.pth并放在**weights**文件夹，然后就可以开始训练了。
## 训练获取自己数据集的权重
在**train_FFA.py**文件中修改数据集和标注的路径，以及修改保存可视化和权重的路径和初始权重的路径，同时根据需要修改**batch size、num epochs**等<br>
`python train_FFA.py`
## 测试
`python demo/test_ap_on_coco_stamenplus.py \
  --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py \
  --checkpoint_path weights/your_weights.pth \
  --image_dir ./stamen_dataset0/images/stamen_test \
  --anno_path ./stamen_dataset0/test_coco_stamen.json \
  --device cuda \
  --num_select 100
![image](https://github.com/user-attachments/assets/3bddc9f6-5759-406e-8172-8dbf89d6d49c)
`
## 精度
工业大麻雄蕊数据在**原groundingdino**项目训练后结果为**AP=0.245**:`python train.py`<br>
而经过**创新点**改后测试发现**AP=0.47**，有明显改进和巨大潜力`python train_FFA.py`


表头  | 表头  | 表头
 ---- | ----- | ------  
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.264
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.470
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.256
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.264
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.032
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.256
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.509
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.509
