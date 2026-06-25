import torch
import torch.nn as nn
from typing import override
from PIL import Image

from src.config import Config, Backbone
from src.encoders.perception_encoder import PerceptionEncoder
from src.heads.head import HeadOutput, LinearProbe
from src.model.base import BaseDeepakeDetectionModel, OutputsForMetrics
from src.utils import logger

class Perception(BaseDeepakeDetectionModel):
    def __init__(self, config: Config):
        super().__init__(config, verbose=True)
        
        # On utilise le backbone spécifié dans la config
        backbone_name = config.backbone if "perception" in config.backbone.lower() else Backbone.PerceptionEncoder_L_p14_336.value
        
        img_size = config.backbone_args.img_size if config.backbone_args else None
        self.feature_extractor = PerceptionEncoder(backbone_name, img_size=img_size)
        
        features_dim = self.feature_extractor.get_features_dim()
        self.head = LinearProbe(features_dim, config.num_classes)
        
        self.criterion = nn.CrossEntropyLoss()

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        features = self.feature_extractor(inputs)
        return self.head(features)

    def _shared_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        loss = self.criterion(outputs.logits_labels, batch.labels)
        return loss, outputs, batch

    @override
    def training_step(self, batch, batch_idx):
        loss, outputs, batch = self._shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True)
        
        probs = outputs.logits_labels.softmax(dim=1)
        self.train_step_outputs.labels.update(batch.labels)
        self.train_step_outputs.probs.update(probs.detach())
        self.train_step_outputs.idx.update(batch.idx)
        return loss

    @override
    def validation_step(self, batch, batch_idx):
        loss, outputs, batch = self._shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True)
        
        probs = outputs.logits_labels.softmax(dim=1)
        self.val_step_outputs.labels.update(batch.labels)
        self.val_step_outputs.probs.update(probs.detach())
        self.val_step_outputs.idx.update(batch.idx)
        return loss

    @override
    def test_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        probs = outputs.logits_labels.softmax(dim=1)

        self.test_step_outputs.labels.update(batch.labels)
        self.test_step_outputs.probs.update(probs.detach())
        self.test_step_outputs.idx.update(batch.idx)

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.config.lr, 
            weight_decay=self.config.weight_decay
        )
        if self.config.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.max_epochs
            )
            return [optimizer], [scheduler]
        return optimizer

    @override
    def load_checkpoint(self, checkpoint_path: str):
        if not checkpoint_path:
            return
        logger.print_info(f"Loading Perception weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '').replace('backbone.', 'feature_extractor.backbone.')
            new_key = new_key.replace('head.', 'head.linear.')
            new_state_dict[new_key] = value
        incompatible_keys = self.load_state_dict(new_state_dict, strict=False)
        self.print_checkpoint_keys(incompatible_keys)

    @override
    def get_preprocessing(self):
        def preprocess(image: Image.Image) -> torch.Tensor:
            image = self.custom_preprocessing(image)
            return self.feature_extractor.preprocess(image)
        return preprocess