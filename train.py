import os
import csv
import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import roc_auc_score

from model import Identity, FeatureExtractor, build_fast_flow
from utils import calc_loss, get_score

torch.backends.cudnn.benchmark = True

class Trainer:
    FeatureShape = {
        "cait_m48_448": (768, 28, 28),
        "deit_base_distilled_patch16_384": (768, 24, 24)
    }

    def __init__(self, ac, config):
        self.cfg = config
        assert self.cfg.backbone in ["cait_m48_448", "deit_base_distilled_patch16_384"], "{} is not implemented.".format(self.cfg.backbone)
        self.feat_shape = self.FeatureShape[self.cfg.backbone]
        self.ac = ac

        self.encoder = self.load_encoder()
        self.cut_tail()
        self.flow = build_fast_flow(self.cfg.clamp, self.cfg.clamp_activation, encoded_shape=self.feat_shape)
        
        self.losses = [0] * self.cfg.nb_epoch
        self.l1s = [0] * self.cfg.nb_epoch
        self.l2s = [0] * self.cfg.nb_epoch
        self.val_normal = []
        self.val_anomaly = []
        self.det_aurocs = []
        self.seg_aurocs = []

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

    def save_weights(self, path):
        torch.save(self.flow.state_dict(), path)

    def load_weights(self, path):
        assert hasattr(self, "flow"), "has no flow yet"
        self.flow.load_state_dict(torch.load(path))

    def plot_metrics(self):
        # plot log loss
        plt.figure()    
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(self.losses[:self.current_epoch+1], label="total")
        plt.plot(self.l1s[:self.current_epoch+1], label="^2")
        plt.plot(self.l2s[:self.current_epoch+1], label="j")
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(self.cfg.result_path, "{}_train_log_loss.png".format(self.ac)))
        plt.close()

        plt.figure()    
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(self.losses[:self.current_epoch+1], label="total")
        plt.plot(self.l1s[:self.current_epoch+1], label="^2")
        plt.plot(self.l2s[:self.current_epoch+1], label="j")
        plt.legend()
        plt.savefig(os.path.join(self.cfg.result_path, "{}_train_loss.png".format(self.ac)))
        plt.close()

        x = [(i+1)*self.cfg.validate_per_epoch for i in range(len(self.val_normal))]
        # plot val loss
        if not self.val_normal:
            return
        plt.figure()    
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(x, self.val_normal, label="normal")
        plt.plot(x, self.val_anomaly, label="anomaly")
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(self.cfg.result_path, "{}_val_loss.png".format(self.ac)))
        plt.close()
        
        # plot auroc
        plt.figure()    
        plt.xlabel("epoch")
        plt.ylabel("auroc")
        plt.plot(x, self.seg_aurocs, label="seg")
        plt.plot(x, self.det_aurocs, label="det")
        plt.legend()
        plt.savefig(os.path.join(self.cfg.result_path, "{}_auroc.png".format(self.ac)))
        plt.close()

    def save_res(self):
        with open("{}/{}_train.csv".format(self.cfg.result_path, self.ac), "w") as f:
            writer = csv.writer(f)
            writer.writerow(self.losses)
            writer.writerow(self.l1s)
            writer.writerow(self.l2s)
            writer.writerow(self.val_normal)
            writer.writerow(self.val_anomaly)
            writer.writerow(self.det_aurocs)
            writer.writerow(self.seg_aurocs)

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
        

    def warmup(self, data_loader):
        if self.cfg.warmup_epoch == 0:
            return
        steps = len(data_loader) * self.cfg.warmup_epoch
        from_lr = self.cfg.learning_rate / 30
        lr_diff = (self.cfg.learning_rate - from_lr) / steps
        lr = from_lr
        for e in range(self.cfg.warmup_epoch):
            with tqdm(data_loader, desc="[Warmup Epoch {}]".format(e+1)) as pbar:
                for x in pbar:
                    self.optimizer.zero_grad()
                    for g in self.optimizer.param_groups:
                        g['lr'] = lr
                    with torch.no_grad():
                        _ = self.encoder(x)
                    feature = torch.permute(self.extractor.saved_feature, (0, 2, 1)).view(-1, *self.feat_shape)
                    z, jac = self.flow(feature)
                    loss, l1, l2 = calc_loss(z, jac)
                    loss = loss.mean() / (self.feat_shape[0] * self.feat_shape[1] * self.feat_shape[2])
                    loss.backward()
                    self.optimizer.step()
                    pbar.set_postfix({
                        "loss": loss.item(), 
                        "lr": lr
                    })
                    lr += lr_diff
        for g in self.optimizer.param_groups:
            g['lr'] = self.cfg.learning_rate

    def train_meta_epoch(self, data_loader, epoch):
        t_loss = 0
        t_l1 = 0
        t_l2 = 0
        with tqdm(data_loader, desc="[Train Epoch {}]".format(epoch+1)) as pbar:
            for x in pbar:
                self.optimizer.zero_grad()
                with torch.no_grad():
                    _ = self.encoder(x)
                feature = torch.permute(self.extractor.saved_feature, (0, 2, 1)).view(-1, *self.feat_shape)
                z, jac = self.flow(feature)
                loss, l1, l2 = calc_loss(z, jac)
                loss = loss.mean() / (self.feat_shape[0] * self.feat_shape[1] * self.feat_shape[2])
                l1 = l1.mean() / (self.feat_shape[0] * self.feat_shape[1] * self.feat_shape[2])
                l2 = l2.mean() / (self.feat_shape[0] * self.feat_shape[1] * self.feat_shape[2])
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({
                    "loss": loss.item(), 
                    "lr": self.optimizer.param_groups[0]['lr']
                })
                t_loss += loss.item()
                t_l1 += l1.item()
                t_l2 += l2.item()
        self.losses[epoch] = t_loss / len(data_loader)
        self.l1s[epoch] = t_l1 / len(data_loader)
        self.l2s[epoch] = t_l2 / len(data_loader)

    def calc_images_loss(self, images):
        nb_iter = -(-len(images) // 32)
        t_loss = 0
        for i in range(nb_iter):
            l = i * self.cfg.batch_size
            r = (i+1) * self.cfg.batch_size
            if r > len(images):
                r = len(images)
            with torch.no_grad():
                self.encoder(torch.stack(images[l:r]).to(self.device))
                feature = torch.permute(self.extractor.saved_feature, (0, 2, 1)).view(-1, *self.feat_shape)
                z, jac = self.flow(feature)
                loss, l1, l2 = calc_loss(z, jac)
                loss = loss.mean() / (self.feat_shape[0] * self.feat_shape[1] * self.feat_shape[2])
                t_loss += loss
        return t_loss / nb_iter

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

    def get_detection_auroc(self, preds, mask):
        image_score = np.max(preds.numpy(), axis=(1, 2, 3))
        label = np.max(mask, axis=(1, 2, 3))
        auroc = roc_auc_score(label, image_score)
        return auroc

    def get_segmentation_auroc(self, preds, masks):
        pixel_socre = self.upsample(preds, size=self.d.original_image_size)
        auroc = roc_auc_score(masks.flatten(), pixel_socre.flatten())
        return auroc

    def calc_auroc(self, dataset):
        self.d = dataset
        images, mask = dataset.load_test()
        mask = torch.stack(mask).numpy()
        pred = self.pred(images)
        pred_min, pred_max = torch.min(pred), torch.max(pred)
        pred = (pred - pred_min) / (pred_max - pred_min)
        detection_auroc = self.get_detection_auroc(pred, mask)
        segmentation_auroc = self.get_segmentation_auroc(pred, mask)
        return detection_auroc, segmentation_auroc

    def run(self, dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True
        )
        self.optimizer = torch.optim.Adam(self.flow.parameters(), lr=self.cfg.learning_rate, weight_decay=1e-5)
        self.warmup(data_loader)

        anomaly_images, anomaly_masks = dataset.load_test(skip_normal=True)
        normal_images, normal_masks = dataset.load_test(only_normal=True)
        for e in range(self.cfg.nb_epoch):
            self.current_epoch = e
            self.train_meta_epoch(data_loader, e)
            self.plot_metrics()
            if ((e+1) % self.cfg.validate_per_epoch) == 0:
                anomaly_loss = self.calc_images_loss(anomaly_images)
                normal_loss = self.calc_images_loss(normal_images)
                self.val_anomaly.append(anomaly_loss.detach().cpu().item())
                self.val_normal.append(normal_loss.detach().cpu().item())
                print("Validation Loss: Anomaly {:.5f}, Normal {:.5f}".format(anomaly_loss, normal_loss))
                det_auroc, seg_auroc = self.calc_auroc(dataset)
                print("Detection AUROC {:.5f}, Segmentation AUROC {:.5f}".format(det_auroc, seg_auroc))
                self.det_aurocs.append(det_auroc)
                self.seg_aurocs.append(seg_auroc)
        self.save_res()