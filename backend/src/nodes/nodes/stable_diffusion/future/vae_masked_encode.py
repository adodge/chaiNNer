# from __future__ import annotations
#
# import numpy as np
# import torch
# from comfy.latent_image import RGBImage, GreyscaleImage
#
# from nodes.impl.stable_diffusion.types import LatentImage, VAEModel, array_to_image
# from nodes.node_base import NodeBase
# from nodes.node_factory import NodeFactory
# from nodes.properties.inputs import ImageInput
# from nodes.properties.inputs.stable_diffusion_inputs import VAEModelInput
# from nodes.properties.outputs.stable_diffusion_outputs import LatentImageOutput
# from nodes.nodes.stable_diffusion import category as StableDiffusionCategory
#
#
# @NodeFactory.register("chainner:stable_diffusion:vae_masked_encode")
# class VAEMaskedEncodeNode(NodeBase):
#     def __init__(self):
#         super().__init__()
#         self.description = ""
#         self.inputs = [
#             ImageInput(channels=3),
#             ImageInput(channels=1),
#             VAEModelInput(),
#         ]
#         self.outputs = [
#             LatentImageOutput(),
#         ]
#
#         self.category = StableDiffusionCategory
#         self.name = "VAE Masked Encode"
#         self.icon = "PyTorch"
#         self.sub = "Input & Output"
#
#     @torch.no_grad()
#     def run(self, image: np.ndarray, mask: np.ndarray, vae: VAEModel) -> LatentImage:
#         try:
#             vae.cuda()
#             img = RGBImage.from_image(array_to_image(image), device="cuda")
#             msk = GreyscaleImage.from_image(array_to_image(mask), device="cuda")
#             latent = vae.masked_encode(img, msk)
#         finally:
#             vae.cpu()
#         return latent.cpu()
