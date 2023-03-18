from nodes.impl.stable_diffusion.types import LatentImage, StableDiffusionModel, CLIPModel

from .. import expression
from .base_output import BaseOutput, OutputKind


class StableDiffusionModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "StableDiffusionModel",
        label: str = "Model",
        kind: OutputKind = "generic",
        should_broadcast=False,
    ):
        super().__init__(model_type, label, kind=kind)
        self.should_broadcast = should_broadcast

    def get_broadcast_data(self, value: StableDiffusionModel):
        if not self.should_broadcast:
            return None

        return {
            "arch": value.version.value,
        }


class CLIPModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "CLIPModel",
        label: str = "CLIP",
        kind: OutputKind = "generic",
        should_broadcast=False,
    ):
        super().__init__(model_type, label, kind=kind)
        self.should_broadcast = should_broadcast

    def get_broadcast_data(self, value: CLIPModel):
        if not self.should_broadcast:
            return None

        return {
            "arch": value.version.value,
        }


class VAEModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "VAEModel",
        label: str = "VAE",
        kind: OutputKind = "generic",
        should_broadcast=False,
    ):
        super().__init__(model_type, label, kind=kind)
        self.should_broadcast = should_broadcast


class ConditioningOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "Conditioning",
        label: str = "Conditioning",
        kind: OutputKind = "generic",
        should_broadcast=False,
    ):
        super().__init__(model_type, label, kind=kind)
        self.should_broadcast = should_broadcast

    def get_broadcast_data(self, value: CLIPModel):
        if not self.should_broadcast:
            return None

        return {
            "arch": value.version.value,
        }


class LatentImageOutput(BaseOutput):
    def __init__(
        self,
        image_type: expression.ExpressionJson = "LatentImage",
        label: str = "Latent",
        kind: OutputKind = "generic",
        should_broadcast=False,
    ):
        super().__init__(image_type, label, kind=kind)
        self.should_broadcast = should_broadcast

    def get_broadcast_data(self, value: LatentImage):
        if not self.should_broadcast:
            return None

        w, h = value.size()

        return {
            "height": h*64,
            "width": w*64,
        }