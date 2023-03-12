# from __future__ import annotations
#
# import os
# from typing import Optional, Tuple
#
# from nodes.impl.stable_diffusion.types import CLIPModel
# from nodes.node_base import NodeBase
# from nodes.node_factory import NodeFactory
# from nodes.properties.inputs import DirectoryInput, SliderInput, StableDiffusionPtFileInput
# from nodes.properties.outputs import DirectoryOutput, FileNameOutput
# from nodes.properties.outputs.stable_diffusion_outputs import CLIPModelOutput
# from nodes.utils.utils import split_file_path
# from nodes.nodes.stable_diffusion import category as StableDiffusionCategory
#
#
# @NodeFactory.register("chainner:stable_diffusion:load_clip")
# class LoadModelNode(NodeBase):
#     def __init__(self):
#         super().__init__()
#         self.description = ""
#         self.inputs = [
#             StableDiffusionPtFileInput(primary_input=True),
#             DirectoryInput("Embedding Directory").make_optional(),
#             SliderInput("Stop At Layer", default=-1, minimum=-24, maximum=-1),
#         ]
#         self.outputs = [
#             CLIPModelOutput(),
#             DirectoryOutput("Model Directory", of_input=0),
#             FileNameOutput("Model Name", of_input=0),
#         ]
#
#         self.category = StableDiffusionCategory
#         self.name = "Load CLIP"
#         self.icon = "PyTorch"
#         self.sub = "Input & Output"
#
#     def run(
#         self, path: str, embedding_directory: Optional[str], stop_at_layer: int
#     ) -> Tuple[CLIPModel, str, str]:
#         assert os.path.exists(path), f"Model file at location {path} does not exist"
#         assert os.path.isfile(path), f"Path {path} is not a file"
#
#         if embedding_directory is not None:
#             assert os.path.exists(
#                 embedding_directory
#             ), f"Embedding directory {embedding_directory} does not exist"
#             assert os.path.isdir(
#                 embedding_directory
#             ), f"Path {embedding_directory} is not a directory"
#
#         clip = CLIPModel.from_model(
#             path,
#             stop_at_clip_layer=stop_at_layer,
#             embedding_directory=embedding_directory,
#         )
#
#         dirname, basename, _ = split_file_path(path)
#         return clip, dirname, basename
