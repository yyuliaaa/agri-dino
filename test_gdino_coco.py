import json
import os
import argparse
import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from groundingdino.util.train import load_model
from tqdm import tqdm
from torch import nn

# 加载模型和权重
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights_x/groundingdino_swint_ogc.pth")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.eval()  # 设置为评估模式
model.to(device)  # 将模型移动到 GPU 或 CPU

def compute_similarity(obj_feats, roi_feats):
    roi_feats = roi_feats.unsqueeze(-2)
    sim = torch.nn.functional.cosine_similarity(roi_feats, obj_feats, dim=-1)
    return sim

# 定义图像预处理函数
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).unsqueeze(0).to(device)
    return img


# 定义特征提取函数
def extract_features(img, model):
    with torch.no_grad():
        features = model(img)
    avg_feature = features.mean(dim=[2, 3])
    return avg_feature


# 生成proposal
def get_object_proposal(image_path, bboxs, ratio=1.0, save_rois=True, output_dir='object_proposals'):
    raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_height, image_width = raw_image.shape[:-1]
    scene_name = os.path.basename(image_path).split('.')[0]
    sel_rois = []

    for bbox in bboxs:
        x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cropped_img = raw_image[y0:y1, x0:x1]
        cropped_img = Image.fromarray(cropped_img)

        # save roi region
        if save_rois:
            os.makedirs(os.path.join(output_dir, scene_name), exist_ok=True)
            cropped_img.save(os.path.join(output_dir, scene_name, f"{scene_name}_{x0}_{y0}_{x1}_{y1}.png"))

        # save bbox
        sel_roi = dict()
        sel_roi['image_id'] = int(scene_name.split('_')[-1])
        sel_roi['bbox'] = [x0, y0, x1 - x0, y1 - y0]
        sel_roi['image_width'] = image_width
        sel_roi['image_height'] = image_height
        sel_rois.append(sel_roi)

    return sel_rois


# 测试图像的特征生成和COCO预测标注生成
def generate_coco_predictions(test_image_folder, object_features, num_object):
    scene_features_list = []
    proposals_list = []

    for image_path in tqdm(os.listdir(test_image_folder)):
        if image_path.lower().endswith('.jpg'):
            image_path = os.path.join(test_image_folder, image_path)
            # 假设您有一个函数获取GroundingDINO模型的精确边界框
            from test_coco_nids import get_bbox_from_gdino
            accurate_bboxs = get_bbox_from_gdino(image_path, model)
            accurate_bboxs = accurate_bboxs.cpu().numpy()
            sel_rois = get_object_proposal(image_path, accurate_bboxs, ratio=1.0, save_rois=False,
                                           output_dir=args.output_dir)

            scene_features = []
            for bbox in accurate_bboxs:
                # 根据bbox裁剪图像
                x0, y0, x1, y1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                cropped_img = Image.open(image_path).crop((x0, y0, x1, y1))
                feature = extract_features(preprocess_image(cropped_img), model)
                scene_features.append(feature)

            scene_features = torch.cat(scene_features, dim=0)
            scene_features = nn.functional.normalize(scene_features, dim=1, p=2)

            scene_features_list.append(scene_features)
            proposals_list.append(sel_rois)

    # 计算相似度和生成COCO预测标注的代码将在这里添加

    return proposals_list  # 返回proposals列表


test_image_folder = 'path_to_your_test_image_folder'
output_dir = 'path_to_your_output_folder'
num_object = 1

# 加载训练图像特征
with open('object_features.json', 'r') as f:
    feat_dict = json.load(f)
object_features = torch.Tensor(feat_dict['features'])

# 生成COCO预测标注
coco_predictions = generate_coco_predictions(test_image_folder, object_features, num_object)

# 保存预测结果
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, "coco_instances_results.json"), "w") as f:
    json.dump(coco_predictions, f)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 加载COCO ground truth注释
cocoGt = COCO('path_to_your_coco_ground_truth.json')

# 加载预测结果
cocoDt = cocoGt.loadRes('path_to_your_coco_instances_results.json')

# 创建COCO评估对象
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

# 评估
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

# 输出结果
print("AP: ", cocoEval.stats[0])  # Average Precision
print("AP50: ", cocoEval.stats[1])  # Average Precision at IoU=0.5
print("AP75: ", cocoEval.stats[2])  # Average Precision at IoU=0.75
print("APs: ", cocoEval.stats[3])  # Average Precision for small objects
print("APm: ", cocoEval.stats[4])  # Average Precision for medium objects
print("APl: ", cocoEval.stats[5])  # Average Precision for large objects
print("AR1: ", cocoEval.stats[6])  # Average Recall at IoU=1
print("AR10: ", cocoEval.stats[7])  # Average Recall at IoU=0.1
print("AR100: ", cocoEval.stats[8])  # Average Recall at IoU=0.01
print("ARs: ", cocoEval.stats[9])  # Average Recall for small objects
print("ARm: ", cocoEval.stats[10])  # Average Recall for medium objects
print("ARl: ", cocoEval.stats[11])  # Average Recall for large objects