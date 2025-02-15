# ------------------------------------------------------------------------
# Grounding DINO. Midified by Shilong Liu.
# url: https://github.com/IDEA-Research/GroundingDINO
# Copyright (c) 2023 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO evaluator that works in distributed mode.

Mostly copy-paste from https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
The difference is that there is less copy-pasting from pycocotools
in the end of the file, as python3 can suppress prints with contextlib
"""
import contextlib
import copy
import os

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from groundingdino.util.misc import all_gather

class CocoGroundingEvaluator(object):
    def __init__(self, coco_gt, iou_types, useCats=True):
        self.coco_results = []
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            self.coco_eval[iou_type].useCats = useCats

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}
        self.useCats = useCats

    def update(self, predictions):
        img_ids = list(np.unique(list(predictions.keys())))
        print('1111111111111')
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            # 确保结果被添加到self.coco_results列表中
            self.coco_results.extend(self.prepare_for_coco_detection(predictions))

            # suppress pycocotools prints
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = COCO.loadRes(self.coco_gt, results) if results else COCO()

            coco_eval = self.coco_eval[iou_type]

            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            coco_eval.params.useCats = self.useCats
            img_ids, eval_imgs = evaluate(coco_eval)
            for image_id in img_ids:
                annotation_ids = self.coco_gt.getAnnIds(imgIds=image_id)
                print(f"----Image ID: {image_id}, Annotation IDs: {annotation_ids}-----")

            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type])

    def accumulate(self):
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            print("-------------------")
            print("image_ids value:", original_id)
            if boxes:  # 如果boxes列表不为空，打印具体的每个框坐标信息
                print("Boxes coordinates details:")
                for box_index, box in enumerate(boxes):
                    print(f"    - Box index: {box_index}")
                    print(f"      x: {box[0]}, y: {box[1]}, width: {box[2]}, height: {box[3]}")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def draw_boxes(self, image, boxes, color=(0, 0, 255), thickness=2):
        """在图像上绘制多个边界框"""
        for box in boxes:
            if len(box) == 4:  # 确保box有四个元素
                x, y, w, h = box
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    def save_detection_results(self, image_dir, result_dir):
        # 确保结果目录存在
        os.makedirs(result_dir, exist_ok=True)

        # 为每个图像绘制所有预测框和标注框并保存
        for image_id in set([result["image_id"] for result in self.coco_results]):
            img_info = self.coco_gt.loadImgs(image_id)[0]
            img_path = os.path.join(image_dir, img_info['file_name'])
            img = cv2.imread(img_path)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            print(f"当前处理的image_id: {image_id}")

            # 获取该图像的所有预测框
            prediction_boxes = [result["bbox"] for result in self.coco_results if result["image_id"] == image_id]
            # 绘制所有预测的边界框（红色）
            self.draw_boxes(img, prediction_boxes, color=(0, 0, 255), thickness=2)

            # 获取该图像的所有标注框id
            annotation_ids = self.coco_gt.getAnnIds(imgIds=image_id)
            print(f"获取到的对应标注框id列表（annotation_ids）：{annotation_ids}")
            # 加载标注信息
            annotations = self.coco_gt.loadAnns(annotation_ids)
            print(f"加载的标注信息（annotations）如下：")
            print(annotations)

            annotation_boxes = []  # 移动到这里，确保只初始化一次

            # 检查iscrowd数据类型及过滤逻辑
            print(f"开始检查iscrowd过滤逻辑，原始标注框数量：{len(annotations)}")
            for annotation in annotations:
                print(f"当前标注框信息：{annotation}")
                print(f"iscrowd的数据类型：{type(annotation['iscrowd'])}")
                # 过滤掉iscrowd为1的标注框，确保数据类型正确比较
                if isinstance(annotation['iscrowd'], int) and annotation['iscrowd'] == 0:
                    x, y, w, h = annotation['bbox']
                    print(f"提取的坐标信息：x={x}, y={y}, w={w}, h={h}")
                    annotation_boxes.append([x, y, w, h])
                    print(f"当前已添加标注框列表（annotation_boxes）：{annotation_boxes}")
                else:
                    print(f"该标注框iscrowd不为0或者数据类型不符，跳过此框。")
            print(f"经过iscrowd过滤后标注框数量：{len(annotation_boxes)}")

            # 绘制标注的边界框（绿色）
            self.draw_boxes(img, annotation_boxes, color=(0, 255, 0), thickness=2)

            # 保存图像
            save_path = os.path.join(result_dir, f"{image_id:012d}.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"Image saved to: {save_path}")
    def prepare_for_coco_segmentation(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0] 
                for mask in masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    all_img_ids = all_gather(img_ids)
    all_eval_imgs = all_gather(eval_imgs)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # keep only unique (and in sorted order) images
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################


def evaluate(self):
    """
    Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
    :return: None
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print("useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType))
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId) 
        for imgId in p.imgIds 
        for catId in catIds}

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet) 
        for catId in catIds 
        for areaRng in p.areaRng 
        for imgId in p.imgIds
    ]
    # this is NOT in the pycocotools code, but could be done outside
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
