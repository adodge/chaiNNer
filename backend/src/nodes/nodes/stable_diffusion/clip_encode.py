from __future__ import annotations

from typing import Tuple, Optional

from ...impl.stable_diffusion.types import CLIPModel, Conditioning

from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import TextAreaInput
from ...properties.inputs.stable_diffusion_inputs import CLIPModelInput
from . import category as StableDiffusionCategory
from ...properties.outputs.stable_diffusion_outputs import ConditioningOutput


@NodeFactory.register("chainner:stable_diffusion:clip_encode")
class CLIPEncodeNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [CLIPModelInput(),
            TextAreaInput("Prompt").make_optional(),]
        self.outputs = [
            ConditioningOutput(),
        ]

        self.category = StableDiffusionCategory
        self.name = "CLIP Encode"
        self.icon = "PyTorch"
        self.sub = "Conditioning"

    def run(self, clip: CLIPModel, prompt: Optional[str]) -> Conditioning:
        prompt = prompt or ""
        out = clip.encode(prompt)
        return out