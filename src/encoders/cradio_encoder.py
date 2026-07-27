import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class CRadioEncoder(nn.Module):
    def __init__(self, model_name="nvidia/C-RADIOv4-SO400M", img_size: None | int = 224):
        """
        Model: nvidia/C-RADIOv4-SO400M | 2304 features (summary)
        """
        super().__init__()

        self._preprocess = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = model_name
        self.features_dim = 2304
        self.img_size = img_size if img_size is not None else 224

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        if self.img_size is not None:
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, preprocessed_images: torch.Tensor) -> torch.Tensor:
        return self.backbone(preprocessed_images).summary

    def get_features_dim(self) -> int:
        return self.features_dim


if __name__ == "__main__":
    import autorootcwd  # noqa: F401

    from src.config import Backbone
    from src.encoders._common import inference

    model = CRadioEncoder(Backbone.C_RADIOV4_SO400M.value)
    inference(model)
