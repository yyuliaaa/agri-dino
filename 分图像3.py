import os
import shutil

# 定义源文件夹路径
source_folder = r'E:\Gong\ultralytics-main\data\images_enhance'

# 定义目标文件夹路径
train_folder = r'E:\Gong\new\Grounding-Dino-FineTuning-main0\stamen_dataset0\images\train_s'
val_folder = r'E:\Gong\new\Grounding-Dino-FineTuning-main0\stamen_dataset0\images\val_s'
test_folder = r'E:\Gong\new\Grounding-Dino-FineTuning-main0\stamen_dataset0\images\test_s'

# 创建目标文件夹如果它们不存在
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# 定义val和test的文件名前缀
val_prefixes = {'DJI_0552', 'DJI_0579', 'DJI_0570', 'DJI_0558', 'DJI_0609', 'DJI_0535'}
test_prefixes = {'DJI_0576', 'DJI_0540', 'DJI_0561', 'DJI_0546', 'DJI_0539', 'DJI_0543', 'DJI_0545'}

# 定义train的文件名（不包括编号）
train_filenames = {
    'DJI_0568', 'DJI_0577', 'DJI_0537', 'DJI_0555', 'DJI_0598', 'DJI_0566', 'DJI_0617', 'DJI_0613',
    'DJI_0554', 'DJI_0559', 'DJI_0556', 'DJI_0614', 'DJI_0615', 'DJI_0536', 'DJI_0549', 'DJI_0533',
    'DJI_0563', 'DJI_0608', 'DJI_0562', 'DJI_0542', 'DJI_0569', 'DJI_0610', 'DJI_0538', 'DJI_0606',
    'DJI_0574', 'DJI_0548', 'DJI_0541', 'DJI_0565', 'DJI_0611', 'DJI_0571', 'DJI_0550', 'DJI_0567',
    'DJI_0572', 'DJI_0551', 'DJI_0547', 'DJI_0553', 'DJI_0544', 'DJI_0534', 'DJI_0560', 'DJI_0616',
    'DJI_0599', 'DJI_0607', 'DJI_0618', 'DJI_0573', 'DJI_0612', 'DJI_0575', 'DJI_0557'
}

# 遍历源文件夹中的所有文件
for filename in os.listdir(source_folder):
    # 提取文件的基础名（不含编号）
    parts = filename.split('_')
    if len(parts) > 1:
        base_filename = '_'.join(parts[:-1])  # 保持文件名格式，不包括最后一个部分（编号）

        # 检查文件是否是val
        if any(val_prefix in filename for val_prefix in val_prefixes):
            shutil.move(os.path.join(source_folder, filename), val_folder)
        # 检查文件是否是test
        elif any(test_prefix in filename for test_prefix in test_prefixes):
            shutil.move(os.path.join(source_folder, filename), test_folder)
        # 检查文件是否是train
        elif base_filename in train_filenames:
            shutil.move(os.path.join(source_folder, filename), train_folder)
        else:
            print(f"File {filename} does not match any category and was not moved.")
    else:
        print(f"File {filename} has an unexpected format and was not moved.")

print("Image classification complete.")