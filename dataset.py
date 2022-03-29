import os
from glob import glob

import torch
from torch.utils.data import Dataset
import torchvision.io as tvio
import torchvision.transforms as T

class MVTec(Dataset):
    CATEGORY = [
        "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", 
        "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"
    ]
    ImageSize = {
        "cait_m48_448": 448,
        "deit_base_distilled_patch16_384": 384
    }

    def __init__(self, category, config, device=None, preload=True):
        self.cfg = config # MVTec dataset folder path
        assert self.cfg.mvtec_path is not None, "need to fill in mvtec_path in config.py"
        self.device = device
        self.category = category
        self.path = os.path.join(self.cfg.mvtec_path, self.category)
        self.train_pathes = glob(os.path.join(self.path, "train/good/*.png"))
        self.anomaly_category = os.listdir(os.path.join(self.path, "test"))
        self.original_image_size = tvio.read_image(self.train_pathes[0]).shape[1]
        self.image_size = self.ImageSize[self.cfg.backbone]
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.Lambda(lambda im: im / 255.0)
        ])
        if preload:
            self.preload()

    def load_trans(self, path):
        return self.transform(tvio.read_image(path)).float()

    def preload(self):
        self.train_image = [self.load_trans(p) for p in self.train_pathes]
        if self.device:
            for i in range(len(self.train_image)):
                self.train_image[i] = self.train_image[i].to(self.device)
        shape = self.train_image[0].shape
        if shape[0] != 3:
            self.train_image = [im.expand(3, -1, -1) for im in self.train_image]
        self.loaded = True

    def __len__(self):
        return len(self.train_pathes)

    def __getitem__(self, idx):
        if self.loaded:
            return self.loaded_read(idx)
        else:
            return self.read(idx)

    def read(self, idx):
        return self.transform(tvio.read_image(self.pathes[idx])).float()

    def loaded_read(self, idx):
        return self.train_image[idx]

    def load_test(self, skip_normal=False, only_normal=False):

        if only_normal:
            image_pathes = glob(os.path.join(self.path, "test/good/*.png"))
            images, masks = self.load_normal(image_pathes)
            return images, masks

        images = [] # [n, 3, 448, 448]
        masks = [] # [n, 1, 448, 448]

        for ac in self.anomaly_category:
            if ac == "good" and skip_normal:
                continue
            image_pathes = glob(os.path.join(self.path, "test/{}/*.png".format(ac)))

            if ac == "good":
                loaded_images, loaded_masks = self.load_normal(image_pathes)
            else:
                image_name = [p.split("/")[-1].split(".")[0] for p in image_pathes]
                mask_pathes = [os.path.join(self.path, "ground_truth", ac, n + "_mask.png") for n in image_name]
                loaded_images, loaded_masks = self.load_anomaly(image_pathes, mask_pathes)
            images.extend(loaded_images)
            masks.extend(loaded_masks)
        return images, masks

    def load_anomaly(self, image_pathes, mask_pathes):
        images = []
        masks = []
        for p in image_pathes:
            images.append(self.load_trans(p))
        for p in mask_pathes:
            masks.append(tvio.read_image(p))
        shape = images[0].shape
        if shape[0] != 3:
            images = [im.expand(3, -1, -1) for im in images]
        return images, masks

    def load_normal(self, pathes):
        images = []
        masks = []
        for p in pathes:
            images.append(self.load_trans(p))
            masks.append(torch.zeros((1, self.original_image_size, self.original_image_size)))
        shape = images[0].shape
        if shape[0] != 3:
            images = [im.expand(3, -1, -1) for im in images]
        return images, masks
