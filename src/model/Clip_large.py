import torch
import torch.nn as nn
from typing import override
from PIL import Image

from src.config import Config, Backbone
from src.encoders.clip_encoder import CLIPEncoder
from src.heads.head import HeadOutput, LinearProbe
from src.model.base import BaseDeepakeDetectionModel, OutputsForMetrics
from src.utils import logger

class CLIP(BaseDeepakeDetectionModel):
    def __init__(self, config: Config):
        super().__init__(config, verbose=True)
        
        # On utilise le backbone spécifié dans la config (par défaut CLIP Large 14)
        backbone_name = config.backbone if "clip" in config.backbone.lower() else Backbone.CLIP_L_14.value
        self.feature_extractor = CLIPEncoder(backbone_name)
        
        # La dimension des features dépend du modèle CLIP choisi (1024 pour Large, 768 pour Base)
        features_dim = self.feature_extractor.get_features_dim()
        
        # On utilise le head LinearProbe standard du projet
        self.head = LinearProbe(features_dim, config.num_classes)
        
        self.test_step_outputs = OutputsForMetrics()

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        # Extraction des features via le CLIPEncoder (utilise pooler_output)
        features = self.feature_extractor(inputs)
        # Passage dans la couche de classification
        return self.head(features)

    @override
    def test_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        outputs = self.forward(batch.images)
        probs = outputs.logits_labels.softmax(dim=1)

        self.test_step_outputs.labels.update(batch.labels)
        self.test_step_outputs.probs.update(probs.detach())
        self.test_step_outputs.idx.update(batch.idx)

    @override
    def load_checkpoint(self, checkpoint_path: str):
        """Chargement robuste des poids compatible avec DeepFakeBench et ce benchmark."""
        if not checkpoint_path:
            return
            
        logger.print_info(f"Loading CLIP weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Nettoyage des préfixes DDP/DeepFakeBench
            new_key = new_key.replace('module.', '').replace('backbone.', 'feature_extractor.vision_model.')
            # Adaptation du nom de la couche de classification vers LinearProbe
            new_key = new_key.replace('head.', 'head.linear.')
            
            new_state_dict[new_key] = value

        incompatible_keys = self.load_state_dict(new_state_dict, strict=False)
        self.print_checkpoint_keys(incompatible_keys)

    @override
    def get_preprocessing(self):
        def preprocess(image: Image.Image) -> torch.Tensor:
            # Applique le redimensionnement/zoom custom si défini dans la config
            image = self.custom_preprocessing(image)
            # Applique le preprocessing spécifique à CLIP (Normalize/Resize)
            return self.feature_extractor.preprocess(image)
        return preprocess