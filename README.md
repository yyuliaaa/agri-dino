# agri-dino（求star星星版）
项目代码见master，数据使用的是工业大麻雄蕊图片，标注文件已放在stamen_dataset文件夹，在stamen_dataset内新建**images**文件夹，内部放置自己的数据集（train、val、test），下载*groundingdino官网*的初始权重groundingdino_swint_ogc.pth并放在**weights**文件夹，然后就可以开始训练了。
## 训练获取自己数据集的权重
在**train.py**文件中修改数据集和标注的路径，以及修改保存可视化和权重的路径和初始权重的路径，同时根据需要修改**batch size、num epochs**等<br>
`python train.py`
## 测试
`python demo/test_ap_on_coco_stamenplus.py --config_file groundingdino/config/GroundingDINO_SwinT_OGC.py --checkpoint_path weights/训练后的权重 --device cuda --anno_path ./stamen_dataset0/test_coco_stamen.json --image_dir E:\Gong\new\Grounding-Dino-FineTuning-main0\stamen_dataset0\images\stamen_test --num_workers 0 --num_select 100`
## 精度
工业大麻雄蕊数据在**原groundingdino**项目训练后结果为**AP=0.107**<br>
而经过**创新点**改后测试发现**AP=0.264**，有明显改进和巨大潜力
