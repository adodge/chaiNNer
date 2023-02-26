from .base_output import BaseOutput, OutputKind
from .. import expression


class StableDiffusionModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "StableDiffusionModel",
        label: str = "Model",
        kind: OutputKind = "generic"
    ):
        super().__init__(model_type, label, kind=kind)


class CLIPModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "CLIPModel",
        label: str = "CLIP",
        kind: OutputKind = "generic"
    ):
        super().__init__(model_type, label, kind=kind)


class VAEModelOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "VAEModel",
        label: str = "VAE",
        kind: OutputKind = "generic"
    ):
        super().__init__(model_type, label, kind=kind)


class ConditioningOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "Conditioning",
        label: str = "Conditioning",
        kind: OutputKind = "generic"
    ):
        super().__init__(model_type, label, kind=kind)


class LatentImageOutput(BaseOutput):
    def __init__(
        self,
        model_type: expression.ExpressionJson = "LatentImage",
        label: str = "Latent",
        kind: OutputKind = "generic"
    ):
        super().__init__(model_type, label, kind=kind)
