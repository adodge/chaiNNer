# from __future__ import annotations
#
# from nodes.impl.stable_diffusion.types import Conditioning
# from nodes.node_base import NodeBase
# from nodes.node_factory import NodeFactory
# from nodes.properties.inputs.stable_diffusion_inputs import ConditioningInput
# from nodes.properties.outputs.stable_diffusion_outputs import ConditioningOutput
# from nodes.nodes.stable_diffusion import category as StableDiffusionCategory
#
#
# @NodeFactory.register("chainner:stable_diffusion:conditioning_compose")
# class ConditinoingComposeNode(NodeBase):
#     def __init__(self):
#         super().__init__()
#         self.description = ""
#         self.inputs = [
#             ConditioningInput(),
#             ConditioningInput(),
#         ]
#         self.outputs = [
#             ConditioningOutput(),
#         ]
#
#         self.category = StableDiffusionCategory
#         self.name = "Compose"
#         self.icon = "PyTorch"
#         self.sub = "Conditioning"
#
#     def run(self, a: Conditioning, b: Conditioning) -> Conditioning:
#         return Conditioning.combine([a, b])
