import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig

# from torchvision.datasets import CocoDetection
import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_predictions(image_path, predictions, category_dict):
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建一个matplotlib图形
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # 遍历预测结果
    for score, label, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
        print(f"Score: {score.item()}, Label: {label.item()}, Box: {box.tolist()}")
        if label.item() in category_dict:  # 确保标签在类别字典中
            category_name = category_dict[label.item()]['name']
            # 绘制边界框
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            # 添加标签
            ax.text(box[0], box[1], f"{category_name} {score:.2f}", color='red', verticalalignment='top',
                    bbox={'color': 'red', 'pad': 0})

    plt.axis('off')
    plt.show()
    plt.savefig("output_image.png")  # 保存图像到文件
    plt.close()  # 关闭图形

def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    state_dict = clean_state_dict(checkpoint['model_state_dict'])
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.images = self.coco.imgs  # 添加这行代码来初始化 images 属性

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: dict

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target


class PostProcessCocoGrounding(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, coco_api=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist)  # 80, 256. normed

        id_map = {0: 0, 1: 1}

        # build a mapping from label_id to pos_map
        new_pos_map = torch.zeros((91, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        pos_maps = self.positive_map.to(prob_to_token.device)
        # (bs, 100, 256) @ (91, 256).T -> (bs, 100, 91)
        prob_to_label = prob_to_token @ pos_maps.T

        # if os.environ.get('IPDB_SHILONG_DEBUG', None) == 'INFO':
        #     import ipdb; ipdb.set_trace()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results


def print_coco_category_ids(dataset):
    # 打印COCO数据集的类别ID和名称
    print("COCO Dataset Category IDs and Names:")
    for category in dataset.coco.dataset['categories']:
        print(f"COCO Category ID: {category['id']}, Name: {category['name']}")

def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()

    # build dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        coco_api=dataset.coco, tokenlizer=tokenlizer)

    # build evaluator
    evaluator = CocoGroundingEvaluator(
        dataset.coco, iou_types=("bbox",), useCats=True)

    # build captions
    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # run inference
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        # feed to the model
        outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)
        results = postprocessor(outputs, orig_target_sizes)

        # Visualize predictions
        for j, result in enumerate(results):
            image_id = targets[j]['image_id']  # 获取图像的 ID
            image_info = dataset.coco.loadImgs(image_id)[0]  # 使用图像 ID 获取图像信息
            image_path = os.path.join(args.image_dir, image_info['file_name'])
            visualize_predictions(image_path, result, category_dict)

        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on COCO", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # coco info
    parser.add_argument("--anno_path", type=str,
                        required=True, help="coco root")
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    args = parser.parse_args()

    main(args)
