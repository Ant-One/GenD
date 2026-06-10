import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import override
from PIL import Image

from src.config import Config
from src.heads.head import HeadOutput
from src.model.base import BaseDeepakeDetectionModel, OutputsForMetrics
from src.model.Xception import XceptionBackbone, preprocessing_xception
from src.utils import logger

class SPSL(BaseDeepakeDetectionModel):
    def __init__(self, config: Config):
        super().__init__(config, verbose=True)
        
        # On utilise le backbone Xception de base
        self.model = XceptionBackbone(num_classes=config.num_classes)
        
        # MODIFICATION CRUCIALE : SPSL prend 4 canaux en entrée (RGB + Phase)
        # On remplace la première couche conv1 (3 -> 32) par (4 -> 32)
        self.model.conv1 = nn.Conv2d(4, 32, 3, 2, 0, bias=False)
        
        self.test_step_outputs = OutputsForMetrics()

    def phase_without_amplitude(self, img: torch.Tensor) -> torch.Tensor:
        """Extrait l'information de phase uniquement (Spatial-Phase Shallow Learning)."""
        # Passage en niveaux de gris (moyenne des canaux RGB)
        gray_img = torch.mean(img, dim=1, keepdim=True) # (B, 1, H, W)
        
        # Transformée de Fourier 2D
        X = torch.fft.fftn(gray_img, dim=(-1, -2))
        
        # Extraction de la phase
        phase_spectrum = torch.angle(X)
        
        # Reconstruction avec magnitude constante (1.0)
        reconstructed_X = torch.exp(1j * phase_spectrum)
        
        # Transformée inverse pour revenir dans le domaine spatial
        reconstructed_x = torch.real(torch.fft.ifftn(reconstructed_X, dim=(-1, -2)))
        return reconstructed_x

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        # 1. Calcul de la composante de phase
        phase_fea = self.phase_without_amplitude(inputs)
        
        # 2. Concaténation RGB + Phase -> 4 canaux
        # inputs: (B, 3, 256, 256), phase_fea: (B, 1, 256, 256)
        x = torch.cat((inputs, phase_fea), dim=1) # (B, 4, 256, 256)
        
        # 3. Passage dans le backbone Xception modifié
        logits = self.model(x)
        return HeadOutput(logits_labels=logits)

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
        """Chargement robuste gérant la conversion 3-canaux vers 4-canaux si nécessaire."""
        if not checkpoint_path: return
        
        logger.print_info(f"Loading SPSL weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('module.', '').replace('backbone.', '')
            
            # Gestion du reshape pointwise (spécifique aux poids Xception de DeepFakeBench)
            if 'pointwise' in new_key and value.ndim == 2:
                value = value.unsqueeze(-1).unsqueeze(-1)
            
            # Gestion spéciale de conv1.weight si on charge un modèle pré-entraîné ImageNet (3ch)
            if new_key == 'conv1.weight' and value.shape[1] == 3:
                logger.print_info("Converting 3-channel conv1 to 4-channel for SPSL...")
                avg_weight = value.mean(dim=1, keepdim=True)
                value = avg_weight.repeat(1, 4, 1, 1)
                
            new_state_dict[new_key] = value

        # On réutilise la logique de Xception pour adjust_channel si présente dans les poids
        if 'last_linear.weight' in new_state_dict:
            in_features = new_state_dict['last_linear.weight'].shape[1]
            if in_features == 512:
                self.model.last_linear = nn.Linear(512, self.model.num_classes)
                self.model.adjust_channel = nn.Sequential(
                    nn.Conv2d(2048, 512, 1, 1),
                    nn.BatchNorm2d(512),
                    nn.ReLU(inplace=False),
                )

        incompatible_keys = self.model.load_state_dict(new_state_dict, strict=False)
        self.print_checkpoint_keys(incompatible_keys)

    @override
    def get_preprocessing(self):
        def preprocess(image: Image.Image) -> torch.Tensor:
            image = self.custom_preprocessing(image)
            return preprocessing_xception(image)
        return preprocess
