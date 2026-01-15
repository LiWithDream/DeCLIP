import json
import logging
import os
import random
import math
from dataclasses import dataclass
from multiprocessing import Value
from typing import List
import numpy as np
from training.misc import get_tokenizer
from training.utils import mask2box
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from open_clip.transform import FixedSizeCrop, _convert_to_rgb, det_image_transform, get_scale
from pycocotools.coco import COCO
from training.coco_api import COCOPanoptic
from panopticapi import utils
from torchvision.transforms import Normalize
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor, Resize, \
    CenterCrop
# import mmcv
import io
# from mmengine.fileio import get
try:
    from petrel_client.client import Client
except:
    Client = None
from open_clip.transform import ResizeLongest



class ProposalDistillDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root,
                 crop_size=224,
                 tokenizer=None, args=None):
        logging.debug(f'Loading coco style data from {input_filename}.')
        self.coco = COCO(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = tokenizer
        self.image_root = image_root
        self.image_ids = list(self.coco.imgs.keys())
        self.max_anns = 20
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self.args = args
        self.min_size = args.min_size
        self.max_size = args.max_size
        self.ceph_root = args.train_ceph_root
        self.use_ceph = (self.ceph_root != "")
        self.FILE_CLIENT = None
        L = args.det_image_size//args.downsample_factor
        if args.use_vfm == "dino-B-8":  # patch 8
            proxy_resolution = L * 8 
        elif args.use_vfm in ["dinov2-L","dinov2-B"]: # patch 14
            proxy_resolution = L* 14
        elif args.use_vfm in ["sam-B","sam-L","dino-B-16"]: # patch 16
            proxy_resolution = L* 16
        else:
            raise NotImplementedError(f"Proxy type '{args.use_vfm}' is not implemented.")
        self.proxy_transform = det_image_transform(
                proxy_resolution,
                is_train=False,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            try:
                img_bytes = self.FILE_CLIENT.get(image_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        else:
            image_path = os.path.join(self.image_root, image_name)
            try:
                image = Image.open(image_path)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        width, height = image.size
        if width < 10 or height < 10:
            print(f"Invalid image, size {image.size}", flush=True)
            return None

        return image

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        if 'file_name' in image_info:
            image_name = image_info['file_name']
        else:
            assert 'coco_url' in image_info
            coco_url = image_info['coco_url'].split('/')
            image_name = os.path.join(coco_url[-2], coco_url[-1])

        old_image = self.read_image(image_name)
        proxy_image=self.proxy_transform(old_image)
        if old_image is None:
            next_id = random.choice(range(self.__len__()))
            return self.__getitem__(next_id)
        img_w, img_h = old_image.width, old_image.height
        new_image = self.transforms[0](old_image)
        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 1)    # xyxy s
        texts=[]
        image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
        indices = list(range(len(anns)))
        random.shuffle(indices)
        num_valid_boxes = 0
        for i, ann_id in enumerate(indices[:self.max_anns]):
            ann = anns[ann_id]
            x, y, w, h = ann['bbox']
            if w*h < (self.min_size ** 2) or w*h > (self.max_size ** 2):
                continue
            num_valid_boxes += 1
            cx, cy = x + w*0.5, y + h*0.5
            x0, y0, x1, y1 = \
                max(cx - w*0.75, 0), max(cy - h*0.75, 0), min(cx + w*0.75, img_w), min(cy + h*0.75, img_h)
            image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))   # image crops
            box_info = torch.tensor([x, y, x + w, y + h, 1.0])    # x, y, x + w, y + h
            boxes_template[i] = box_info

        if num_valid_boxes == 0:
            boxes_template[0] = torch.tensor([0, 0, img_w / 4, img_h / 4, 1.0])    # avoid empty
            image_crops[0] = self.transforms[1](old_image.crop((0, 0, img_w // 4, img_h // 4)))

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h
        return new_image, boxes_template, image_crops,proxy_image
    

class GridDistillDataset(Dataset):
    def __init__(self,
                 input_filename,
                 transforms,
                 image_root,
                 max_split=16,
                 crop_size=224,
                 pre_transforms=False,
                 ceph_root="",
                 args=None):
        if os.path.basename(input_filename) in ['lvis_v1_train.json', 'instances_train2017.json']:
            # coco style distillation
            logging.debug(f'Loading coco style data from {input_filename}.')
            self.coco = COCO(input_filename)
            logging.debug('Done loading data.')
            image_ids = list(self.coco.imgs.keys())
            self.style="coco"
        elif os.path.basename(input_filename) in ['chat.json','mixed_data.json','llava_v1_5_mix624k.json']:
            # llava style distillation
            with open(input_filename, 'r') as file:
                data = json.load(file)
            image_ids = [item["image"] for item in data]
            self.style="llava"
        else:
            raise ValueError(f"Unsupported file format or style for {input_filename}.")
        self._init_choices(max_split)
        self.transforms = transforms
        self.image_root = image_root
        self.args = args
        train_ratio = args.train_ratio
        if train_ratio < 1.0:
            num_images = int(len(image_ids) * train_ratio)
            random.shuffle(image_ids)
            image_ids = image_ids[:num_images]
        self.image_ids = image_ids
        self.max_anns = args.max_boxes
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self._init_boxes()
        self.ceph_root = ceph_root
        self.use_ceph = (ceph_root != "")
        self.FILE_CLIENT = None
        L = args.det_image_size//args.downsample_factor
        if args.use_vfm:
            if args.use_vfm == "dino-B-8":  # patch 8
                proxy_resolution = L * 8 
            elif args.use_vfm in ["dinov2-L","dinov2-B"]: # patch 14
                proxy_resolution = L* 14
            elif args.use_vfm in ["sam-B","sam-L","dino-B-16"]: # patch 16
                proxy_resolution = L* 16
            else:
                raise NotImplementedError(f"Proxy type '{args.use_vfm}' is not implemented.")
            self.proxy_transform = det_image_transform(
                    proxy_resolution,
                    is_train=False,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
        else:
            self.proxy_transform=None

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            try:
                img_bytes = self.FILE_CLIENT.get(image_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        else:
            image_path = os.path.join(self.image_root, image_name)
            try:
                image = Image.open(image_path)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None

        width, height = image.size
        if width < 10 or height < 10:
            print(f"Invalid image, size {image.size}", flush=True)
            return None
        return image

    def _init_choices(self, M=16):
        choices = []
        for m in range(1, M+1):
            for n in range((m + 1)//2, min(m*2 + 1, M+1)):
                choices.append((m, n))
        self.choices = choices

    def __len__(self):
        return len(self.image_ids)

    def _init_boxes(self, ):
        box_templates = {}
        for choice in self.choices:
            M, N = choice
            grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, N + 1), torch.linspace(0, 1, M + 1),
                                            indexing='xy')
            x0y0s = torch.stack([grid_x[:M, :N], grid_y[:M, :N]], dim=-1)
            x1y1s = torch.stack([grid_x[1:, 1:], grid_y[1:, 1:]], dim=-1)
            pseudo_boxes = torch.cat([x0y0s, x1y1s],dim=-1).view(-1, 4)

            assert pseudo_boxes.shape[0] == M*N
            box_templates[choice] = pseudo_boxes

        self.box_templates = box_templates

    @staticmethod
    def _bbox_iou(box1, box2):
        x0 = max(box1[0], box2[0])
        y0 = max(box1[1], box2[1])
        x1 = min(box1[2], box2[2])
        y1 = min(box1[3], box2[3])
        inter_w = max(0.0, x1 - x0)
        inter_h = max(0.0, y1 - y0)
        inter_area = inter_w * inter_h
        if inter_area <= 0:
            return 0.0
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        denom = area1 + area2 - inter_area
        if denom <= 0:
            return 0.0
        return inter_area / denom

    def _get_output_size(self):
        output_size = self.args.input_size
        if isinstance(output_size, (tuple, list)):
            output_size = output_size[0]
        return int(output_size)

    def _get_patch_size(self):
        model_name = getattr(self.args, "model", "").lower()
        if "16" in model_name:
            return 16
        if "14" in model_name:
            return 14
        return 16

    @staticmethod
    def _random_crop_box(img_w, img_h, min_scale=0.4, max_scale=0.9):
        crop_w = max(2, int(img_w * random.uniform(min_scale, max_scale)))
        crop_h = max(2, int(img_h * random.uniform(min_scale, max_scale)))
        if img_w - crop_w <= 0 or img_h - crop_h <= 0:
            return 0.0, 0.0, float(img_w), float(img_h)
        x0 = random.randint(0, img_w - crop_w)
        y0 = random.randint(0, img_h - crop_h)
        return float(x0), float(y0), float(x0 + crop_w), float(y0 + crop_h)

    def _sample_overlap_pair(self, img_w, img_h, min_iou):
        max_attempts = 10
        for _ in range(max_attempts):
            box1 = self._random_crop_box(img_w, img_h)
            box1_w = box1[2] - box1[0]
            box1_h = box1[3] - box1[1]
            jitter_x = int(random.uniform(-0.3, 0.3) * box1_w)
            jitter_y = int(random.uniform(-0.3, 0.3) * box1_h)
            x0 = min(max(0, int(box1[0] + jitter_x)), img_w - int(box1_w))
            y0 = min(max(0, int(box1[1] + jitter_y)), img_h - int(box1_h))
            box2 = float(x0), float(y0), float(x0 + box1_w), float(y0 + box1_h)
            if self._bbox_iou(box1, box2) >= min_iou:
                return box1, box2
        return box1, box1

    @staticmethod
    def _map_to_output_coords(crop_w, crop_h, x0, y0, x1, y1, output_size):
        scale = output_size / max(crop_h, crop_w)
        new_w = int(round(crop_w * scale))
        new_h = int(round(crop_h * scale))
        pad_w = output_size - new_w
        pad_h = output_size - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        x0_out = x0 * scale + pad_left
        x1_out = x1 * scale + pad_left
        y0_out = y0 * scale + pad_top
        y1_out = y1 * scale + pad_top
        return x0_out, y0_out, x1_out, y1_out

    def _compute_overlap_meta(self, box1, box2, output_size, patch_size):
        crop1_w = box1[2] - box1[0]
        crop1_h = box1[3] - box1[1]
        crop2_w = box2[2] - box2[0]
        crop2_h = box2[3] - box2[1]
        inter_x0 = max(box1[0], box2[0])
        inter_y0 = max(box1[1], box2[1])
        inter_x1 = min(box1[2], box2[2])
        inter_y1 = min(box1[3], box2[3])
        if inter_x1 <= inter_x0 or inter_y1 <= inter_y0:
            grid_size = torch.tensor([output_size // patch_size, output_size // patch_size], dtype=torch.long)
            return {
                "v1_start": torch.zeros(2, dtype=torch.long),
                "v2_start": torch.zeros(2, dtype=torch.long),
                "overlap_size": torch.zeros(2, dtype=torch.long),
                "grid_size": grid_size,
                "valid": torch.tensor(False, dtype=torch.bool),
            }
        inter1_x0 = inter_x0 - box1[0]
        inter1_y0 = inter_y0 - box1[1]
        inter1_x1 = inter_x1 - box1[0]
        inter1_y1 = inter_y1 - box1[1]
        inter2_x0 = inter_x0 - box2[0]
        inter2_y0 = inter_y0 - box2[1]
        inter2_x1 = inter_x1 - box2[0]
        inter2_y1 = inter_y1 - box2[1]

        out1 = self._map_to_output_coords(crop1_w, crop1_h, inter1_x0, inter1_y0, inter1_x1, inter1_y1, output_size)
        out2 = self._map_to_output_coords(crop2_w, crop2_h, inter2_x0, inter2_y0, inter2_x1, inter2_y1, output_size)

        grid_h = output_size // patch_size
        grid_w = output_size // patch_size

        v1_x0 = max(0, min(grid_w, int(out1[0] // patch_size)))
        v1_y0 = max(0, min(grid_h, int(out1[1] // patch_size)))
        v1_x1 = max(0, min(grid_w, int(math.ceil(out1[2] / patch_size))))
        v1_y1 = max(0, min(grid_h, int(math.ceil(out1[3] / patch_size))))

        v2_x0 = max(0, min(grid_w, int(out2[0] // patch_size)))
        v2_y0 = max(0, min(grid_h, int(out2[1] // patch_size)))
        v2_x1 = max(0, min(grid_w, int(math.ceil(out2[2] / patch_size))))
        v2_y1 = max(0, min(grid_h, int(math.ceil(out2[3] / patch_size))))

        overlap_h = min(v1_y1 - v1_y0, v2_y1 - v2_y0)
        overlap_w = min(v1_x1 - v1_x0, v2_x1 - v2_x0)
        if overlap_h <= 0 or overlap_w <= 0:
            grid_size = torch.tensor([grid_h, grid_w], dtype=torch.long)
            return {
                "v1_start": torch.zeros(2, dtype=torch.long),
                "v2_start": torch.zeros(2, dtype=torch.long),
                "overlap_size": torch.zeros(2, dtype=torch.long),
                "grid_size": grid_size,
                "valid": torch.tensor(False, dtype=torch.bool),
            }

        return {
            "v1_start": torch.tensor([v1_y0, v1_x0], dtype=torch.long),
            "v2_start": torch.tensor([v2_y0, v2_x0], dtype=torch.long),
            "overlap_size": torch.tensor([overlap_h, overlap_w], dtype=torch.long),
            "grid_size": torch.tensor([grid_h, grid_w], dtype=torch.long),
            "valid": torch.tensor(True, dtype=torch.bool),
        }

    def _obtain_image_crops(self, image, choice):
        image_crops = []
        img_w, img_h = image.size
        normed_boxes = self.box_templates[choice]
        indices = list(range(len(normed_boxes)))
        random.shuffle(indices)
        indices = indices[:self.max_anns]
        boxes = normed_boxes * torch.tensor([img_w, img_h, img_w, img_h])
        for idx in indices:
            box = boxes[idx]
            x0, y0, x1, y1 = box.tolist()    # todo expand
            if self.args.crop_scale > 1.0:
                box_w, box_h = x1 - x0, y1 - y0
                cx, cy = (x1 + x0)/2, (y1 + y0)/2
                delta_factor = 0.5 * self.args.crop_scale
                x0, y0, x1, y1 = max(cx - box_w * delta_factor, 0), max(cy - box_h * delta_factor, 0), \
                    min(cx + box_w * delta_factor, img_w), min(cy + box_h * delta_factor, img_h)
            vanilla_view=self.transforms[1](image.crop((x0, y0, x1, y1)))
            image_crops.append(vanilla_view)
        return torch.stack(image_crops), boxes[indices]
    
    def _load_target(self, id: int):
        return self.coco.loadAnns(self.coco.getAnnIds(id))
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        if self.style=="coco":
            image_info = self.coco.imgs[image_id]
            if 'file_name' in image_info:
                image_name = image_info['file_name']
            else:
                assert 'coco_url' in image_info
                coco_url = image_info['coco_url'].split('/')
                image_name = os.path.join(coco_url[-2], coco_url[-1])
        else:
            image_name=image_id
        old_image = self.read_image(image_name)
        if self.proxy_transform:
            proxy_image=self.proxy_transform(old_image)
        else:
            proxy_image = torch.empty(0)
        if old_image is None:
            next_id = random.choice(range(self.__len__()))
            return self.__getitem__(next_id)
        if self.args.use_dcac:
            img_w, img_h = old_image.size
            box1, box2 = self._sample_overlap_pair(img_w, img_h, self.args.overlap_min_iou)
            view1 = self.transforms[1](old_image.crop(box1))
            view2 = self.transforms[1](old_image.crop(box2))
            output_size = self._get_output_size()
            patch_size = self._get_patch_size()
            overlap_meta = self._compute_overlap_meta(box1, box2, output_size, patch_size)
        else:
            view1 = torch.empty(0)
            view2 = torch.empty(0)
            overlap_meta = {
                "v1_start": torch.zeros(2, dtype=torch.long),
                "v2_start": torch.zeros(2, dtype=torch.long),
                "overlap_size": torch.zeros(2, dtype=torch.long),
                "grid_size": torch.zeros(2, dtype=torch.long),
                "valid": torch.tensor(False, dtype=torch.bool),
            }
        new_image = self.transforms[0](old_image)
        scale = get_scale(old_image, new_image)
        boxes_template = torch.zeros(self.max_anns, 4 + 1)   
        image_crops_template = torch.zeros(self.max_anns, 3, *self.crop_size)
        image_crops, boxes = self._obtain_image_crops(old_image,random.choice(self.choices))
        assert image_crops.shape[0] == boxes.shape[0]
        _, h, w = new_image.shape
        boxes[:, :4] *= scale
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        boxes_template[:boxes.shape[0], :4] = boxes
        boxes_template[:boxes.shape[0], 4] = 1.0
        image_crops_template[:boxes.shape[0]] = image_crops
        if self.args.precompute_knn:
            if self.args.use_dcac:
                return new_image, boxes_template, image_crops_template, proxy_image, view1, view2, overlap_meta, image_id
            return new_image, boxes_template, image_crops_template, proxy_image, image_id
        if self.args.use_dcac:
            return new_image, boxes_template, image_crops_template, proxy_image, view1, view2, overlap_meta
        return new_image, boxes_template, image_crops_template, proxy_image
        

class COCOPanopticDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root, embed_path,
                 segm_root,
                 crop_size=224,
                 tokenizer=None,
                 downsample_factor=16,
                 min_size=8, 
                 max_size=1024,
                 args=None):
        logging.debug(f'Loading coco caption style data from {input_filename}.')
        self.coco = COCOPanoptic(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = tokenizer
        self.image_root = image_root
        self.embeddings = np.load(embed_path)
        self.image_ids = list(self.coco.imgs.keys())
        num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
        self.max_anns = min(max(num_annos), 100)
        if not isinstance(crop_size, (tuple, list)):
            crop_size = [crop_size, crop_size]
        self.crop_size = crop_size
        self.min_size = 8  # fix for val
        self.max_size = 1024
        self.segm_root = segm_root
        self.downsample_factor = downsample_factor
        self.segm_transform = ResizeLongest(max_size=self.transforms[0].transforms[0].max_size // downsample_factor,
                                            fill=0)       # downsample to the output size of image encoder
        self.args=args
        cat_ids = sorted([cat['id'] for cat in self.coco.cats.values()])
        self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}
        self.label2cat_id = {label: cat_id for cat_id, label in self.cat_id2label.items()}
    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def _load_segm(segm_path):
        segmentation = np.array(
            Image.open(segm_path),
            dtype=np.uint8
        )
        # img_bytes = get(segm_path)
        # pan_png = mmcv.imfrombytes(
        #     img_bytes, flag='color', channel_order='rgb').squeeze()
        segm_map = utils.rgb2id(segmentation)

        return segm_map

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_name = image_info['file_name']
        segm_file = image_info['segm_file']
        image_path = os.path.join(self.image_root, image_name)
        segm_path = os.path.join(self.segm_root, segm_file)
        segm_map = self._load_segm(segm_path)
        old_image = Image.open(image_path)
        img_w, img_h = old_image.width, old_image.height
        new_image = self.transforms[0](old_image)
        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 2 + 1 + 1)    # xyxy c valid size, isthing
        image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
        gt_masks = torch.zeros(self.max_anns, self.segm_transform.max_size,self.segm_transform.max_size)
        masked_image_crops = torch.zeros(self.max_anns, 3, *self.crop_size)
        for i, ann in enumerate(anns):
            if i == self.max_anns:
                break
            cat_id = ann['category_id']
            is_thing = self.coco.cats[cat_id]['isthing']
            if is_thing > 0:
                x, y, w, h = ann['bbox']
                cx, cy = x + w*0.5, y + h*0.5
                x0, y0, x1, y1 = \
                    max(cx - w*0.75, 0), max(cy - h*0.75, 0), min(cx + w*0.75, img_w), min(cy + h*0.75, img_h)
            else:
                x0, y0, x1, y1 = mask2box(segm_map == ann['id'])
                x, y, w, h = x0, y0, x1 - x0, y1 - y0
            if w * h < (self.min_size ** 2) or w * h > (self.max_size ** 2):
                continue
            image_crops[i] = self.transforms[1](old_image.crop((x0, y0, x1, y1)))   # image crops
            # masked image crop
            np_old_image = np.asarray(old_image.copy())
            np_old_image[segm_map != ann['id']] = 114
            masked_old_image = Image.fromarray(np_old_image)
            masked_image_crops[i] = self.transforms[1](masked_old_image.crop((x0, y0, x1, y1)))   # image crops

            gt_mask = torch.from_numpy(segm_map == ann['id']).float()
            gt_mask = self.segm_transform(gt_mask[None]) > 0.0
            cls_label = self.cat_id2label[cat_id]
            box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0, w * h, is_thing])    # x, y, x + w, y + h
            boxes_template[i] = box_info
            gt_masks[i] = gt_mask[0]
        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h
        return image_name, new_image, boxes_template, image_crops, gt_masks, masked_image_crops


class COCORegionCLIPDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root, args):
        logging.debug(f'Loading coco caption style data from {input_filename}.')
        self.coco = COCO(input_filename)
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.image_root = image_root
        image_ids = list(self.coco.imgToAnns.keys())    # only use images that have anns
        train_ratio = args.train_ratio
        if train_ratio < 1.0:
            num_images = int(len(image_ids) * train_ratio)
            random.shuffle(image_ids)
            image_ids = image_ids[:num_images]
        self.image_ids = image_ids

        num_annos = [len(anns) for anns in self.coco.imgToAnns.values()]
        self.max_anns = min(max(num_annos), 20)
        self.args = args
        self.ceph_root = args.train_ceph_root
        self.use_ceph = (self.ceph_root != "")
        self.FILE_CLIENT = None
        cat_ids = sorted([cat['id'] for cat in self.coco.cats.values()])

        self.cat_id2label = {cat_id: label for label, cat_id in enumerate(cat_ids)}

    def __len__(self):
        return len(self.image_ids)

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            img_bytes = self.FILE_CLIENT.get(image_path)
            buff = io.BytesIO(img_bytes)
            image = Image.open(buff)
        else:
            image_path = os.path.join(self.image_root, image_name)
            image = Image.open(image_path)
        return image

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.imgs[image_id]
        image_name = image_info['file_name']
        # image_path = os.path.join(self.image_root, image_name)
        # old_image = Image.open(image_path)
        old_image = self.read_image(image_name)
        new_image = self.transforms[0](old_image)

        scale = get_scale(old_image, new_image)
        anns = self.coco.imgToAnns[image_id]
        boxes_template = torch.zeros(self.max_anns, 4 + 2)    # xyxy cls valid

        for i, ann in enumerate(anns):
            if i == self.max_anns:
                break
            cat_id = ann['category_id']
            x, y, w, h = ann['bbox']
            cls_label = self.cat_id2label[cat_id]
            box_info = torch.tensor([x, y, x + w, y + h, cls_label, 1.0])    # x, y, x + w, y + h
            boxes_template[i] = box_info

        _, h, w = new_image.shape

        boxes_template[:, :4] *= scale
        boxes_template[:, [0, 2]] /= w
        boxes_template[:, [1, 3]] /= h

        return new_image, boxes_template


class COCOCaptionDataset(Dataset):
    def __init__(self, input_filename, transforms, image_root,
                 tokenizer=None, args=None):
        logging.debug(f'Loading coco caption style data from {input_filename}.')
        with open(input_filename, 'r') as f:
            self.images = json.load(f)['images']
        logging.debug('Done loading data.')
        self.transforms = transforms
        self.tokenize = get_tokenizer(args.model)
        self.image_root = image_root
        self.ceph_root = args.train_ceph_root
        self.use_ceph = (self.ceph_root != "")
        self.FILE_CLIENT = None

    def read_image(self, image_name):
        if self.use_ceph:
            image_path = os.path.join(self.ceph_root, image_name)
            if self.FILE_CLIENT is None:
                self.FILE_CLIENT = Client()
            try:
                img_bytes = self.FILE_CLIENT.get(image_path)
                buff = io.BytesIO(img_bytes)
                image = Image.open(buff)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None
        else:
            image_path = os.path.join(self.image_root, image_name)
            try:
                image = Image.open(image_path)
            except:
                print(f"Cannot load {image_path}", flush=True)
                return None

        width, height = image.size
        if width < 10 or height < 10:
            print(f"Invalid image, size {image.size}", flush=True)
            return None

        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        text = random.choice(image_info['captions'])
        image_name = image_info['file_name']
        image = self.read_image(image_name)
        if image is None:
            next_id = random.choice(range(self.__len__()))
            return self.__getitem__(next_id)
        image = self.transforms(image)
        text = self.tokenize([text])[0]
        return image, text


def get_coco_panoptic_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    input_filename = args.train_data if is_train else args.val_data
    if args.image_crop_size>0 :
        image_crop_size=args.image_crop_size
    else:
        if args.model=="EVA02-CLIP-B-16" or args.model=="ViT-B-16" or args.model=="ViT-L-14":
            image_crop_size=224
        elif args.model=="siglip-so400m-patch14-384":
            image_crop_size=384 
        else:
            image_crop_size=336 # ViT-L-14-336 & EVA02-CLIP-L-14-336
    assert input_filename
    dataset = COCOPanopticDataset(
        input_filename,
        preprocess_fn,
        segm_root=args.val_segm_root,
        image_root=args.val_image_root,
        embed_path=args.embed_path,
        tokenizer=tokenizer,
        crop_size=image_crop_size,
        min_size=args.min_size,
        max_size=args.max_size,
        downsample_factor=args.downsample_factor,
        args=args,
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    if is_train:
        batch_size = args.batch_size
    else:
        batch_size = min(args.batch_size, 1)     # only support bs = 1 for inference
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_proposal_distill_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train
    input_filename = args.train_data  # if is_train else args.val_data
    assert input_filename
    dataset = ProposalDistillDataset(
        input_filename,
        preprocess_fn,
        image_root=args.train_image_root,
        tokenizer=tokenizer,
        crop_size=args.input_size,
        args=args
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_grid_distill_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train
    input_filename = args.train_data
    assert input_filename
    dataset = GridDistillDataset(
        input_filename=input_filename,
        transforms=preprocess_fn,
        image_root=args.train_image_root,
        crop_size=args.input_size,
        max_split=args.max_split,
        ceph_root=args.train_ceph_root,
        pre_transforms=args.pre_transforms,
        args=args
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_region_clip_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train
    input_filename = args.train_data
    assert input_filename
    dataset = COCORegionCLIPDataset(
        input_filename=input_filename,
        transforms=preprocess_fn,
        image_root=args.train_image_root,
        args=args,
    )
    num_samples = len(dataset)
    # TODO: distributed for test
    sampler = DistributedSampler(dataset) if args.distributed else None  #  and is_train else None
    shuffle = is_train and sampler is None
    batch_size = args.batch_size
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


def get_coco_caption_dataset(args, preprocess_fn, is_train, epoch=0, tokenizer=None):
    assert is_train
    input_filename = args.train_data
    assert input_filename
    dataset = COCOCaptionDataset(
        input_filename,
        preprocess_fn,
        image_root=args.train_image_root,
        tokenizer=tokenizer,
        args=args
    )
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def get_dataset_fn(data_path, dataset_type):
    if dataset_type == 'coco_panoptic':
        return get_coco_panoptic_dataset
    elif dataset_type == 'proposals_distill':
        return get_proposal_distill_dataset
    elif dataset_type == 'grid_distill':
        return get_grid_distill_dataset
    elif dataset_type == 'region_clip':
        return get_region_clip_dataset
    elif dataset_type == 'coco_caption':
        return get_coco_caption_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")


def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}
    if args.train_data:
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, dataset_type=args.test_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    return data
