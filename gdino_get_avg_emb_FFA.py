import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import json
from tqdm import trange
from groundingdino.util.train import load_model
from groundingdino.util.vl_utils import build_captions_and_token_span

# 加载模型和权重
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/checkpoint_epoch_250_0.913.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()  # 设置为评估模式
model.to(device)  # 将模型移动到 GPU 或 CPU

def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objects = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append((xmin, ymin, xmax, ymax))
    print(f"BBoxes found in {xml_path}: {objects}")  # 打印边界框信息
    return objects

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0)  # 添加批次维度
    assert img.shape == (1, 3, 448, 448), f"Image shape is incorrect: {img.shape}"
    return img


def extract_features(img, model, device, caption):
    with torch.no_grad():
        # 创建一个包含单个描述的列表
        captions = [caption]  # 直接使用字符串列表
        # 传递图像和标记化的描述信息给模型
        features = model(img.to(device), captions=captions)
    avg_feature = features.mean(dim=[2, 3])
    return avg_feature

def process_images(image_folder, xml_folder, output_dir, json_filename):
    object_features = []
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.jpg', '.JPG')):
            image_path = os.path.join(image_folder, filename)
            xml_filename = filename.replace('.JPG', '.xml').replace('.jpg', '.xml')  # 同时处理两种扩展名
            xml_path = os.path.join(xml_folder, xml_filename)

            if os.path.exists(xml_path):
                bboxes = parse_xml(xml_path)
                print(bboxes)
                for bbox in bboxes:
                    img = preprocess_image(image_path)
                    img_cropped = img[:, :, max(0, bbox[1]):min(bbox[3], img.shape[2]), max(0, bbox[0]):min(bbox[2], img.shape[3])]
                    if img_cropped.shape[2] > 0 and img_cropped.shape[3] > 0:
                        avg_feature = extract_features(img_cropped, model, device, "hemp")
                        if avg_feature is not None and avg_feature.shape[0] > 0:
                            object_features.append(avg_feature.cpu().numpy())
                            print(f"Feature added for {filename}: {avg_feature.shape}")
                        else:
                            print(f"Invalid feature for {filename}")
                    else:
                        print(f"Invalid bbox for {filename}: {bbox}")

    if object_features:
        object_features_tensor = torch.tensor(object_features)
        feat_dict = {'features': object_features_tensor.tolist()}
        with open(os.path.join(output_dir, json_filename), 'w') as f:
            json.dump(feat_dict, f)
        print(f"Features saved to {os.path.join(output_dir, json_filename)}")
    else:
        print("No features were collected.")

# 图像文件夹路径和XML文件夹路径
image_folder = r'E:\Gong\new\Grounding-Dino-FineTuning-main0\stamen_dataset0\images\train'
xml_folder = r'E:\Gong\ultralytics-main\data\Annotations'
output_dir = './out1'
json_filename = 'object_features.json'

process_images(image_folder, xml_folder, output_dir, json_filename)




