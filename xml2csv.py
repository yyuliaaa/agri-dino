import os
import csv
import xml.etree.ElementTree as ET

# 定义输出CSV文件的路径
output_csv_path = r'E:\Gong\new\Grounding-Dino-FineTuning-main0\stamen_dataset0\train_stamen.csv'

# 定义要遍历的目录
annotations_dir = r'E:\Gong\ultralytics-main\data\xml_enhance'

# 定义要提取的信息的字段名
fieldnames = ['label_name', 'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 'image_name', 'width', 'height']

# 创建一个CSV文件，并写入表头
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # 遍历annotations_dir目录下的所有xml文件
    for filename in os.listdir(annotations_dir):
        if filename.endswith('.xml'):  # 确保是xml文件
            # 构建XML文件的完整路径
            xml_path = os.path.join(annotations_dir, filename)

            # 解析XML文件
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 提取图像的宽度和高度
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            image_name = root.find('filename').text

            # 遍历所有的<object>标签
            for obj in root.findall('object'):
                # 提取标签中的信息
                label_name = obj.find('name').text
                bbox = obj.find('bndbox')
                bbox_x = int(bbox.find('xmin').text)
                bbox_y = int(bbox.find('ymin').text)
                bbox_width = int(bbox.find('xmax').text) - bbox_x
                bbox_height = int(bbox.find('ymax').text) - bbox_y

                # 将提取的信息写入CSV文件
                writer.writerow({
                    'label_name': label_name,
                    'bbox_x': bbox_x,
                    'bbox_y': bbox_y,
                    'bbox_width': bbox_width,
                    'bbox_height': bbox_height,
                    'image_name': image_name,
                    'width': width,
                    'height': height
                })

print(f"All annotations have been merged into {output_csv_path}")