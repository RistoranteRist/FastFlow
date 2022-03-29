import torch

from train import Trainer
from dataset import MVTec

import config as cfg

def main():
    assert cfg.weight_path is not None, "need to fill in weight_path in config.py"
    assert cfg.result_path is not None, "need to fill in result_path in config.py"
    det = {}
    seg = {}
    device = torch.device(cfg.device)

    for c in MVTec.CATEGORY:
        print("category {}".format(c))
        dataset = MVTec(c, cfg, device=device)
        trainer = Trainer(c, nb_epochs=cfg.nb_epoch, backbone=cfg.backbone, lr=cfg.learning_rate, batch_size=cfg.batch_size, result_path=cfg.result_path)
        trainer.to_device(device)
        trainer.run(dataset)
        trainer.save_weights("{}/{}_{}.pth".format(cfg.weight_path, c, cfg.backbone))

if __name__ == "__main__":
    main()