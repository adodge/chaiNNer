from nodes.impl.stable_diffusion.types import LatentImage, StableDiffusionModel, CLIPModel

from .. import expression
from .base_output import BaseOutput, OutputKind


class StableDiffusionModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "StableDiffusionModel",
        label: str = "Model",
        kind: OutputKind = "generic",
    ):
        super().__init__(model_type, label, kind=kind)

    def get_broadcast_data(self, value: StableDiffusionModel):
        version = value.version

        return {
            "version": version.value,
        }


class CLIPModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "CLIPModel",
        label: str = "CLIP",
        kind: OutputKind = "generic",
    ):
        super().__init__(model_type, label, kind=kind)

    def get_broadcast_data(self, value: CLIPModel):
        version = value.version

        return {
            "version": version.value,
        }


class VAEModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "VAEModel",
        label: str = "VAE",
        kind: OutputKind = "generic",
    ):
        super().__init__(model_type, label, kind=kind)


class ConditioningOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "Conditioning",
        label: str = "Conditioning",
        kind: OutputKind = "generic",
    ):
        super().__init__(model_type, label, kind=kind)


class LatentImageOutput(BaseOutput):
    def __init__(
        self,
        image_type: expression.ExpressionJson = "LatentImage",
        label: str = "Latent",
        kind: OutputKind = "generic",
    ):
        super().__init__(image_type, label, kind=kind)

    def get_broadcast_data(self, value: LatentImage):
        w,h = value.size()

        return {
            "height": h*64,
            "width": w*64,
        }