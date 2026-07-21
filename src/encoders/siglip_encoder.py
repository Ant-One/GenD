import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, SiglipVisionModel


class SiglipEncoder(nn.Module):
    def __init__(self, model_name="google/siglip2-so400m-patch14-224"):
        """
        Model: google/siglip2-so400m-patch14-224 | 1152 features
        """
        super().__init__()

        self._preprocess = AutoProcessor.from_pretrained(model_name)
        self.vision_model = SiglipVisionModel.from_pretrained(model_name)
        self.model_name = model_name
        self.features_dim = self.vision_model.config.hidden_size

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        return self.vision_model(preprocessed_images).pooler_output

    def get_features_dim(self) -> int:
        return self.features_dim


if __name__ == "__main__":
    import autorootcwd  # noqa: F401

    from src.config import Backbone
    from src.encoders._common import inference

    model = SiglipEncoder(Backbone.SIGLIP2_SO400M_P14_224.value)
    inference(model)
