from ..expression import ExpressionJson
from .base_input import BaseInput


class StableDiffusionModelInput(BaseInput):
    def __init__(
        self, label: str = "Model", input_type: ExpressionJson = "StableDiffusionModel"
    ):
        super().__init__(input_type, label)


class CLIPModelInput(BaseInput):
    def __init__(self, label: str = "CLIP", input_type: ExpressionJson = "CLIPModel"):
        super().__init__(input_type, label)


class VAEModelInput(BaseInput):
    def __init__(self, label: str = "VAE", input_type: ExpressionJson = "VAEModel"):
        super().__init__(input_type, label)


class ConditioningInput(BaseInput):
    def __init__(
        self, label: str = "Conditioning", input_type: ExpressionJson = "Conditioning"
    ):
        super().__init__(input_type, label)


class LatentImageInput(BaseInput):
    def __init__(
        self, label: str = "Latent", input_type: ExpressionJson = "LatentImage"
    ):
        super().__init__(input_type, label)
