import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig

import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator


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

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: list

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
    def __init__(self, num_select=300, coco_api=None, tokenlizer=None):
        super().__init__()
        self.num_select = num_select

        assert coco_api is not None
        category_dict = coco_api.dataset['categories']
        cat_list = [item['name'] for item in category_dict]
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(tokenlizer(captions), tokenspanlist)

        id_map = {0: 0, 1: 1}

        new_pos_map = torch.zeros((91, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map
        self.positive_map = new_pos_map

    def extract_foreground_features(self, features, boxes):
        # features: (batch_size, num_channels, height, width)
        # boxes: (batch_size, num_queries, 4) in xyxy format
        batch_size, num_channels, height, width = features.shape
        features_avg = torch.zeros((batch_size, num_channels), device=features.device)

        for i in range(batch_size):
            for j in range(boxes.shape[1]):
                x1, y1, x2, y2 = boxes[i, j].long()
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
                features_avg[i] += features[i, :, y1:y2, x1:x2].sum(dim=(-2, -1))
            if boxes.shape[1] > 0:
                features_avg[i] /= boxes.shape[1]
        return features_avg

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        prob_to_token = out_logits.sigmoid()
        pos_maps = self.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        return results


def print_coco_category_ids(dataset):
    print("COCO Dataset Category IDs and Names:")
    for category in dataset.coco.dataset['categories']:
        print(f"COCO Category ID: {category['id']}, Name: {category['name']}")


def main(args):
    cfg = SLConfig.fromfile(args.config_file)
    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model.eval()

    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = CocoDetection(args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(coco_api=dataset.coco, tokenlizer=tokenlizer)
    evaluator = CocoGroundingEvaluator(dataset.coco, iou_types=("bbox",), useCats=True)

    category_dict = dataset.coco.dataset['categories']
    cat_list = [item['name'] for item in category_dict]
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    start = time.time()
    for i, (images, targets) in enumerate(data_loader):
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        outputs = model(images, captions=input_captions)
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        pos_maps = postprocessor.positive_map.to(prob_to_token.device)
        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(targets)
        assert targets[0]['orig_size'].shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), postprocessor.num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not postprocessor.not_to_xyxy:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        else:
            boxes = out_bbox

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        img_h, img_w = targets[0]['orig_size'].unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]

        pred_labels = [result['labels'] for result in results]
        for j, labels in enumerate(pred_labels):
            print(f"Image {i + 1}, Predicted Labels: {labels.tolist()}")
        cocogrounding_res = {
            target["image_id"]: output for target, output in zip(targets, results)}
        evaluator.update(cocogrounding_res)

        if (i + 1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i + 1e-5) * used_time - used_time
            print(f"processed {i}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()

    print("Final results:", evaluator.coco_eval["bbox"].stats.tolist())

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Grounding DINO eval on COCO", add_help=True)
    parser.add_argument("--config_file", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file")
    parser.add_argument("--device", type=str, default="cuda", help="running device (default: cuda)")
    parser.add_argument("--num_select", type=int, default=300, help="number of topk to select")
    parser.add_argument("--anno_path", type=str, required=True, help="coco root")
    parser.add_argument("--image_dir", type=str, required=True, help="coco image dir")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")
    args = parser.parse_args()
    main(args)





