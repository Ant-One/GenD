from typing import override
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from src.config import Config
from src.heads.head import HeadOutput
from src.model.base import BaseDeepakeDetectionModel, OutputsForMetrics
from src.utils import logger

# Preprocessing spécifique à Xception (Normalisation vers [-1, 1])
preprocessing_xception = T.Compose(
    [
        T.Resize((256, 256), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(
            [0.5, 0.5, 0.5], 
            [0.5, 0.5, 0.5], 
        ),
    ]
)

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x

class XceptionBackbone(nn.Module):
    def __init__(self, num_classes=2):
        super(XceptionBackbone, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.last_linear = nn.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x
    
    def classifier(self, features):
        # Si on a détecté (lors du load_checkpoint) qu'il faut utiliser adjust_channel
        # car last_linear attend 512 entrées.
        if hasattr(self, 'adjust_channel') and self.last_linear.in_features == 512:
             x = self.adjust_channel(features)
        else:
             x = F.relu(features) # Note: DFB fait un relu ici s'il n'y a pas adjust_channel !
             
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.classifier(x)
        return x

class Xception(BaseDeepakeDetectionModel):
    def __init__(self, config: Config):
        super().__init__(config, verbose=True)
        self.model = XceptionBackbone(num_classes=config.num_classes)
        self.test_step_outputs = OutputsForMetrics()

    @override
    def forward(self, inputs: torch.Tensor) -> HeadOutput:
        logits = self.model(inputs)
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
        """Chargement robuste des poids de DeepFakeBench."""
        logger.print_info(f"Loading DeepFakeBench Xception weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
            
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            
            # 1. Nettoyage des préfixes DDP/Wrapper
            new_key = new_key.replace('module.', '').replace('backbone.', '')
            
            # 2. Reshape pointwise
            if 'pointwise' in new_key and value.ndim == 2:
                value = value.unsqueeze(-1).unsqueeze(-1)
                
            new_state_dict[new_key] = value

        # 3. Vérification de la taille de last_linear dans le checkpoint
        # C'est LA clé du mystère : 2048 ou 512 ?
        if 'last_linear.weight' in new_state_dict:
            weight_shape = new_state_dict['last_linear.weight'].shape
            in_features = weight_shape[1] # Dimension d'entrée
            
            logger.print_info(f"Checkpoint last_linear expects {in_features} input features.")
            
            # Si le checkpoint a été entraîné avec adjust_channel (512 features)
            if in_features == 512:
                logger.print_info("Mode 'adjust_channel' detected. Rebuilding last_linear.")
                # On recrée last_linear pour accepter 512 au lieu de 2048
                self.model.last_linear = nn.Linear(512, self.model.num_classes)
                # On s'assure que adjust_channel est bien dans votre modèle !
                if not hasattr(self.model, 'adjust_channel'):
                    self.model.adjust_channel = nn.Sequential(
                        nn.Conv2d(2048, 512, 1, 1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=False),
                    )
            
            # Si le checkpoint a été entraîné sans adjust_channel (2048 features)
            elif in_features == 2048:
                logger.print_info("Standard mode detected (2048 features). Removing adjust_channel weights.")
                # On supprime les poids d'adjust_channel du state_dict car on ne s'en sert pas
                keys_to_remove = [k for k in new_state_dict.keys() if 'adjust_channel' in k]
                for k in keys_to_remove:
                    del new_state_dict[k]
                    
        # 4. Chargement final
        # On utilise strict=False car on a peut-être supprimé adjust_channel
        incompatible_keys = self.model.load_state_dict(new_state_dict, strict=False)
        self.print_checkpoint_keys(incompatible_keys)

    @override
    def get_preprocessing(self):
        def preprocess(image: Image.Image) -> torch.Tensor:
            # r, g, b = image.split()
            # image = Image.merge("RGB", (b, g, r))
            image = self.custom_preprocessing(image)
            return preprocessing_xception(image)
        return preprocess