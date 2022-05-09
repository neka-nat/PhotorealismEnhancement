import os
import pathlib
import torch
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import numpy as np
from PIL import Image

from torch.utils.data import DataLoader, Dataset


class PipeDatasets(Dataset):
    def __init__(self, data_root: str = "data/pipe/virtual/", input_size=(640, 480), no_label=False) -> None:
        super(PipeDatasets, self).__init__()
        image_path = pathlib.Path(data_root) / "ColorImages"
        label_path = None if no_label else pathlib.Path(data_root) / "SegmentationMaps"
        self._image_files = sorted(list(image_path.iterdir()))
        self._label_files = None
        if label_path is not None:
            self._label_files = sorted(list(label_path.iterdir()))
        self.input_size = input_size

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx):
        image_file = self._image_files[idx]
        image = Image.open(image_file)
        image = image.resize(self.input_size)
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)

        label = []
        if self._label_files is not None:
            label_file = self._label_files[idx]
            label = Image.open(label_file)
            label = label.resize(self.input_size)
            label = label.convert("L")
            label = np.array(label) > 0
            label = torch.from_numpy(label).unsqueeze(0)
        return {"image": image, "mask": label, "name": image_file.name}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
n_cpu = os.cpu_count()

train_dataloader = DataLoader(PipeDatasets(), batch_size=16, shuffle=True, num_workers=n_cpu)
test_dataloader1 = DataLoader(PipeDatasets(), batch_size=10, shuffle=False, num_workers=n_cpu)
test_dataloader2 = DataLoader(PipeDatasets("data/pipe/real/BulkObjectImages", no_label=True), batch_size=10, shuffle=False, num_workers=n_cpu)

class PipeModel(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]
        assert image.ndim == 4
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


model = PipeModel("FPN", "resnet34", in_channels=3, out_classes=1).to(device)

trainer = pl.Trainer(
    gpus=1, 
    max_epochs=30,
)

trainer.fit(
    model, 
    train_dataloaders=train_dataloader, 
)

itr1 = iter(test_dataloader1)
for _ in range(len(test_dataloader1)):
    batch = next(itr1)
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()
    for fname, pr_mask in zip(batch["name"], pr_masks):
        pr_mask_img = Image.fromarray((pr_mask.numpy().squeeze() * 255).astype(np.uint8))
        pr_mask_img.save(os.path.join("results/virtuals", fname))

itr2 = iter(test_dataloader2)
for _ in range(len(test_dataloader2)):
    batch = next(itr2)
    with torch.no_grad():
        model.eval()
        logits = model(batch["image"])
    pr_masks = logits.sigmoid()
    for fname, pr_mask in zip(batch["name"], pr_masks):
        pr_mask_img = Image.fromarray((pr_mask.numpy().squeeze() * 255).astype(np.uint8))
        pr_mask_img.save(os.path.join("results/reals", fname))
