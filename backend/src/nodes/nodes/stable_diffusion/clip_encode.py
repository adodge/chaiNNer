from __future__ import annotations

from typing import Optional

import torch

from ...impl.stable_diffusion.types import CLIPModel, Conditioning
from ...node_base import NodeBase
from ...node_factory import NodeFactory
from ...properties.inputs import TextAreaInput
from ...properties.inputs.stable_diffusion_inputs import CLIPModelInput
from ...properties.outputs.stable_diffusion_outputs import ConditioningOutput
from . import category as StableDiffusionCategory


@NodeFactory.register("chainner:stable_diffusion:clip_encode")
class CLIPEncodeNode(NodeBase):
    def __init__(self):
        super().__init__()
        self.description = ""
        self.inputs = [
            CLIPModelInput(),
            TextAreaInput("Prompt").make_optional(),
        ]
        self.outputs = [
            ConditioningOutput(
                model_type="""Conditioning {
                arch: Input0.arch
            }"""
            ),
        ]

        self.category = StableDiffusionCategory
        self.name = "CLIP Encode"
        self.icon = "PyTorch"
        self.sub = "Conditioning"

    @torch.no_grad()
    def run(self, clip: CLIPModel, prompt: Optional[str]) -> Conditioning:
        prompt = prompt or ""
        try:
            clip.cuda()
            out = clip.encode(prompt)
        finally:
            clip.cuda()
        return out.cpu()
