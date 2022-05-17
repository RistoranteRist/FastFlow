import os
import csv
import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from model import Identity, FeatureExtractor, build_fast_flow
from utils import calc_loss, get_score, load_image

import config as cfg

torch.backends.cudnn.benchmark = True

class Tester:
    FeatureShape = {
        "cait_m48_448": (768, 28, 28),
        "deit_base_distilled_patch16_384": (768, 24, 24)
    }

    def __init__(self, weight_path, config):
        self.cfg = config
        assert self.cfg.backbone in ["cait_m48_448", "deit_base_distilled_patch16_384"], "{} is not implemented.".format(self.cfg.backbone)
        self.feat_shape = self.FeatureShape[self.cfg.backbone]

        self.encoder = self.load_encoder()
        self.cut_tail()
        self.flow = build_fast_flow(self.cfg.clamp, self.cfg.clamp_activation, encoded_shape=self.feat_shape)
        self.load_weights(weight_path)
        
    def load_encoder(self):
        encoder = timm.create_model(self.cfg.backbone, pretrained=True)
        encoder.eval()
        self.extractor = FeatureExtractor(self.cfg.backbone)
        if self.cfg.backbone == "cait_m48_448":
            _ = encoder.blocks[40].register_forward_hook(self.extractor)
        elif self.cfg.backbone == "deit_base_distilled_patch16_384":
            _ = encoder.blocks[7].register_forward_hook(self.extractor)
        return encoder

    def cut_tail(self):
        if self.cfg.backbone == "cait_m48_448":
            for i in range(len(self.encoder.blocks[41:])):
                module_id = i + 41
                self.encoder.blocks[module_id] = Identity()
            for i in range(len(self.encoder.blocks_token_only)):
                self.encoder.blocks_token_only[i] = Identity()
            self.encoder.norm = Identity()
            self.encoder.head = Identity()
        elif self.cfg.backbone == "deit_base_distilled_patch16_384":
            for i in range(len(self.encoder.blocks[8:])):
                module_id = i + 8
                self.encoder.blocks[module_id] = Identity()
            self.encoder.norm = Identity()

    def to_device(self, device):
        self.device = device
        self.encoder.to(device)
        self.flow.to(device)

    def load_weights(self, path):
        assert hasattr(self, "flow"), "has no flow yet"
        self.flow.load_state_dict(torch.load(path))

    def measure_speed(self, data_loader):
        print("speed test")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        warmup = 2

        for _ in range(warmup):
            for x in data_loader:
                torch.cuda.synchronize()
                time.time()
                with torch.no_grad():
                    _ = self.encoder(x)
                    feature = torch.permute(self.extractor.saved_feature, (0, 2, 1)).view(-1, *self.feat_shape)
                    z, jac = self.flow(feature)
                    score = get_score(z, jac)
                torch.cuda.synchronize()

        elapsed_time = []
        epoch = 5
        for _ in range(epoch):
            for x in data_loader:
                torch.cuda.synchronize()
                start = time.time()
                with torch.no_grad():
                    _ = self.encoder(x)
                    feature = torch.permute(self.extractor.saved_feature, (0, 2, 1)).view(-1, *self.feat_shape)
                    z, jac = self.flow(feature)
                    score = get_score(z, jac)
                torch.cuda.synchronize()
                end = time.time()
                elapsed_time.append(end - start)
        print(np.mean(elapsed_time))

    def upsample(self, images, size):
        return F.interpolate(images, size=size, mode="bilinear", align_corners=True)

    def pred(self, images):
        nb_iter = -(-len(images) // self.cfg.batch_size)
        preds = []
        for i in tqdm(range(nb_iter), desc="[test]"):
            l = i * self.cfg.batch_size
            r = (i + 1) * self.cfg.batch_size
            if r > len(images):
                r = len(images)
            with torch.no_grad():
                self.encoder(torch.stack(images[l:r]).to(self.device))
                feature = torch.permute(self.extractor.saved_feature, (0, 2, 1)).view(-1, *self.feat_shape)
                z, jac = self.flow(feature)
                pred = get_score(z, jac) # shape (batch_size, h, w)
            preds.append(pred.cpu())
        preds = torch.concat(preds, axis=0)
        return preds

    def test_image(self, image_path, save_path, device):
        img = load_image(image_path).unsqueeze(0).to(device)
        to_image = T.ToPILImage()
        with torch.no_grad():
            self.encoder(img)
            feature = torch.permute(self.extractor.saved_feature, (0, 2, 1)).view(-1, *self.feat_shape)
            z, jac = self.flow(feature)
            pred = get_score(z, jac).detach()
        pred /= torch.max(pred)
        pred_img = to_image(pred[0])
        pred_img.save(save_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("weight_path", type=str)
    parser.add_argument("image_path", type=str)
    parser.add_argument("save_path", type=str)
    args = parser.parse_args()
    device = torch.device(cfg.device)
    tester = Tester(args.weight_path, cfg)
    tester.to_device(device)
    tester.test_image(args.image_path, args.save_path, device)